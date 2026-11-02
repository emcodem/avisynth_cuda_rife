// MOCK of mvtools_cuda.dll (API v20 as specified in MVTOOLS_CUDA_CHANGES.md) + the two AviSynth filters needed to drive it
// (MockUpload ~ CudaUpload, MockDownload ~ CudaDownload). Test aid only: it implements the registry with plain cudaMallocPitch
// frames and synchronous copies, to exercise mlrt_trt's token-frame path (struct layouts, pitches, headers, ownership).
// It says nothing about the real library's ordering/eviction behaviour.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>

#include <cuda_runtime.h>
#include <avisynth_c.h>

#include "../src/mvcuda_interop.h"

namespace {

struct MockFrame {
    MvCudaFrameFormat fmt{};
    void * p[3]{};
    size_t pitch[3]{};
};

std::mutex g_lock;
std::map<unsigned long long, MockFrame> g_frames;
std::map<unsigned long long, unsigned long long> g_keys;   // content key -> token
unsigned long long g_next = 1;

void plane_dims(const MvCudaFrameFormat & f, int p, int & w, int & h) {
    w = p == 0 ? f.width : (f.width + f.x_ratio_uv - 1) / f.x_ratio_uv;
    h = p == 0 ? f.height : (f.height + f.y_ratio_uv - 1) / f.y_ratio_uv;
}

bool alloc_frame(const MvCudaFrameFormat & f, MockFrame & m) {
    m.fmt = f;
    for (int p = 0; p < f.planes; ++p) {
        int w, h;
        plane_dims(f, p, w, h);
        if (cudaMallocPitch(&m.p[p], &m.pitch[p], static_cast<size_t>(w) * f.bytes_per_sample, h) != cudaSuccess) return false;
    }
    return true;
}

MvCudaTokenHeader make_header(unsigned long long token, const MvCudaFrameFormat & f, unsigned long long producer, int frame) {
    MvCudaTokenHeader h{};
    h.magic = MVCUDA_TOKEN_MAGIC; h.layout = MVCUDA_TOKEN_LAYOUT; h.device = 0; h.token = token;
    h.producer = producer; h.frame = frame; h.format = f;
    return h;
}

} // namespace

extern "C" {
__declspec(dllexport) int mvcuda_api_version() { return 20; }
__declspec(dllexport) int mvcuda_device_count() { return 1; }

__declspec(dllexport) int mvcuda_frame_upload_keyed(const MvCudaFrameFormat * f, const void * const src[3], const int pitch[3],
    unsigned long long key, MvCudaTokenHeader * out) {
    std::lock_guard<std::mutex> lk(g_lock);
    if (auto it = g_keys.find(key); it != g_keys.end() && g_frames.count(it->second)) {
        *out = make_header(it->second, *f, 0, 0);
        return 0;
    }
    MockFrame m;
    if (!alloc_frame(*f, m)) return -2;
    for (int p = 0; p < f->planes; ++p) {
        int w, h;
        plane_dims(*f, p, w, h);
        if (cudaMemcpy2D(m.p[p], m.pitch[p], src[p], pitch[p], static_cast<size_t>(w) * f->bytes_per_sample, h, cudaMemcpyHostToDevice) != cudaSuccess) return -2;
    }
    const auto token = g_next++;
    g_frames[token] = m;
    g_keys[key] = token;
    *out = make_header(token, *f, 0, 0);
    return 0;
}

__declspec(dllexport) int mvcuda_frame_download(unsigned long long token, void * const dst[3], const int pitch[3]) {
    std::lock_guard<std::mutex> lk(g_lock);
    auto it = g_frames.find(token);
    if (it == g_frames.end()) return MVCUDA_TOKEN_GONE;
    cudaDeviceSynchronize();
    const MockFrame & m = it->second;
    for (int p = 0; p < m.fmt.planes; ++p) {
        int w, h;
        plane_dims(m.fmt, p, w, h);
        if (cudaMemcpy2D(dst[p], pitch[p], m.p[p], m.pitch[p], static_cast<size_t>(w) * m.fmt.bytes_per_sample, h, cudaMemcpyDeviceToHost) != cudaSuccess) return -2;
    }
    return 0;
}

__declspec(dllexport) unsigned long long mvcuda_producer_register(MvCudaRegenFn, void *) { return 1; }
__declspec(dllexport) void mvcuda_producer_unregister(unsigned long long) {}
__declspec(dllexport) int mvcuda_token_regenerate(unsigned long long, int, void *, MvCudaTokenHeader *) { return -7; }
__declspec(dllexport) int mvcuda_token_pin(unsigned long long) { return 0; }
__declspec(dllexport) void mvcuda_token_unpin(unsigned long long) {}

__declspec(dllexport) int mvcuda_token_acquire(unsigned long long token, MvCudaFrameFormat * fmt, unsigned long long planes[3],
    int pitch_bytes[3], void * /*wait_stream*/) {
    std::lock_guard<std::mutex> lk(g_lock);
    auto it = g_frames.find(token);
    if (it == g_frames.end()) return MVCUDA_TOKEN_GONE;
    cudaDeviceSynchronize();   // mock: the real library orders with events instead of waiting on the host
    *fmt = it->second.fmt;
    for (int p = 0; p < 3; ++p) {
        planes[p] = reinterpret_cast<unsigned long long>(it->second.p[p]);
        pitch_bytes[p] = static_cast<int>(it->second.pitch[p]);
    }
    return 0;
}
__declspec(dllexport) void mvcuda_token_release(unsigned long long, void *) {}

__declspec(dllexport) int mvcuda_frame_alloc(const MvCudaFrameFormat * f, unsigned long long producer, int frame,
    unsigned long long planes[3], int pitch_bytes[3], MvCudaTokenHeader * out) {
    std::lock_guard<std::mutex> lk(g_lock);
    MockFrame m;
    if (!alloc_frame(*f, m)) return -2;
    const auto token = g_next++;
    g_frames[token] = m;
    for (int p = 0; p < 3; ++p) {
        planes[p] = reinterpret_cast<unsigned long long>(m.p[p]);
        pitch_bytes[p] = static_cast<int>(m.pitch[p]);
    }
    *out = make_header(token, *f, producer, frame);
    return 0;
}
__declspec(dllexport) int mvcuda_frame_commit(unsigned long long, void *) { return 0; }
__declspec(dllexport) void mvcuda_frame_discard(unsigned long long) {}
}

// ---- AviSynth filters -------------------------------------------------------------------------------------------------

static MvCudaFrameFormat format_of(const AVS_VideoInfo * vi) {
    return MvCudaFrameFormat{ vi->width, vi->height, 3, avs_component_size(vi), avs_bits_per_component(vi),
        1 << avs_get_plane_width_subsampling(vi, AVS_PLANAR_U), 1 << avs_get_plane_height_subsampling(vi, AVS_PLANAR_U) };
}

static AVS_VideoFrame * AVSC_CC upload_get_frame(AVS_FilterInfo * fi, int n) {
    AVS_VideoFrame * src = avs_get_frame(fi->child, n);
    const int planes[3] = { AVS_PLANAR_Y, AVS_PLANAR_U, AVS_PLANAR_V };
    const void * sp[3]; int pp[3];
    for (int p = 0; p < 3; ++p) { sp[p] = avs_get_read_ptr_p(src, planes[p]); pp[p] = avs_get_pitch_p(src, planes[p]); }
    const MvCudaFrameFormat f = format_of(&fi->vi);
    MvCudaTokenHeader h{};
    const unsigned long long key = (reinterpret_cast<uintptr_t>(fi) * 1000003ULL) ^ (static_cast<unsigned long long>(n) + 77);
    if (mvcuda_frame_upload_keyed(&f, sp, pp, key, &h) != 0) { avs_release_video_frame(src); return nullptr; }
    AVS_VideoFrame * dst = avs_new_video_frame_p(fi->env, &fi->vi, src);
    mv_write_token(dst, h);
    avs_release_video_frame(src);
    return dst;
}

static AVS_VideoFrame * AVSC_CC download_get_frame(AVS_FilterInfo * fi, int n) {
    AVS_VideoFrame * src = avs_get_frame(fi->child, n);
    MvCudaTokenHeader h{};
    if (!mv_read_token(src, h)) return src;   // normal frame: pass through
    AVS_VideoFrame * dst = avs_new_video_frame_p(fi->env, &fi->vi, src);
    const int planes[3] = { AVS_PLANAR_Y, AVS_PLANAR_U, AVS_PLANAR_V };
    void * dp[3]; int pp[3];
    for (int p = 0; p < 3; ++p) { dp[p] = avs_get_write_ptr_p(dst, planes[p]); pp[p] = avs_get_pitch_p(dst, planes[p]); }
    const int rc = mvcuda_frame_download(h.token, dp, pp);
    avs_release_video_frame(src);
    if (rc != 0) { avs_release_video_frame(dst); return nullptr; }
    return dst;
}

static AVS_Value AVSC_CC create_filter(AVS_ScriptEnvironment * env, AVS_Value args, void * param) {
    AVS_FilterInfo * fi;
    AVS_Clip * clip = avs_new_c_filter(env, &fi, avs_array_elt(args, 0), 1);
    fi->get_frame = param ? upload_get_frame : download_get_frame;
    AVS_Value v = avs_new_value_clip(clip);
    avs_release_clip(clip);
    return v;
}

extern "C" __declspec(dllexport) const char * AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment * env) {
    avs_add_function(env, "MockUpload", "c", create_filter, reinterpret_cast<void *>(1));
    avs_add_function(env, "MockDownload", "c", create_filter, nullptr);
    return "mock mvtools_cuda";
}

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#if not __cpp_lib_atomic_wait
#include <chrono>
#include <thread>
using namespace std::chrono_literals;
#endif

#include <avisynth_c.h>

#include <cuda_runtime.h>
#include <NvInferRuntime.h>

#include "cuda_helper.h"
#include "cuda_utils.h"
#include "trt_utils.h"
#include "inference_helper.h"
#include "mvcuda_interop.h"

#ifdef _WIN32
#include <Windows.h>

static std::wstring translateName(const char * name) noexcept {
    auto size = MultiByteToWideChar(CP_UTF8, 0, name, -1, nullptr, 0);
    std::wstring ret(static_cast<size_t>(size), {});
    MultiByteToWideChar(CP_UTF8, 0, name, -1, ret.data(), size);
    return ret;
}

// Windows resolves a delay-loaded DLL's dependencies (nvinfer_10.dll here) relative to the
// *host application's* directory (ffmpeg.exe, AvsPmod.exe, ...), not this plugin's own directory
// -- so whether it loads depends on what happens to be sitting next to whichever app loaded it.
// Pre-loading nvinfer_10.dll by an absolute path derived from this DLL's own location makes it
// self-sufficient regardless of host, matching mlrt_ort.cpp's LoadLibraryA-by-own-path convention.
static std::optional<std::string> getOwnModuleDirectory() noexcept {
    HMODULE self{};
    if (!GetModuleHandleExA(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
        reinterpret_cast<LPCSTR>(&getOwnModuleDirectory),
        &self
    )) {
        return std::nullopt;
    }

    char path[MAX_PATH];
    DWORD len{ GetModuleFileNameA(self, path, MAX_PATH) };
    if (len == 0 || len == MAX_PATH) {
        return std::nullopt;
    }

    std::string path_str{ path, len };
    auto pos{ path_str.find_last_of("\\/") };
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    return path_str.substr(0, pos);
}

// Loads `dll_name` from the directory this DLL itself lives in (not the host application's
// directory -- see getOwnModuleDirectory()'s comment above). LOAD_WITH_ALTERED_SEARCH_PATH also
// makes the loaded DLL's OWN dependencies resolve from that same directory first.
//
// This covers two distinct problems with one mechanism:
//   1. nvinfer_10.dll is a real MSVC /DELAYLOAD import of this DLL -- without this, its delay
//      load thunk would resolve it (and, if it loaded some other way, its own dependencies)
//      relative to the host app's directory.
//   2. cudart64_13.dll is NOT a PE-level import of this DLL at all (dumpbin /dependents shows
//      nothing) -- NVIDIA's own cudart.lib performs a bare, path-less LoadLibraryA("cudart64_
//      13.dll") internally on first CUDA runtime call, invisible to the PE import table, which
//      hits the exact same host-app-relative-search problem. Explicitly loading it here first
//      means that later bare LoadLibraryA call just finds the already-loaded module by name
//      (Windows does not re-search the filesystem for an already-loaded module), regardless of
//      who triggers it -- our own code, or nvinfer_10.dll's internals.
static bool preloadFromOwnDirectory(const std::string & dll_name, std::string & err_out) noexcept {
    if (GetModuleHandleA(dll_name.c_str())) {
        return true; // already loaded (e.g. host app already has it) -- nothing to do
    }
    auto own_dir{ getOwnModuleDirectory() };
    if (!own_dir.has_value()) {
        err_out = "failed to determine mlrt_trt.dll's own directory while loading " + dll_name;
        return false;
    }
    std::string full_path{ own_dir.value() + "\\" + dll_name };
    if (!LoadLibraryExA(full_path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH)) {
        err_out = "failed to load " + full_path + " (place " + dll_name + " next to mlrt_trt.dll).";
        return false;
    }
    return true;
}

static std::string sanitizeForFilename(std::string s) noexcept {
    for (auto & c : s) {
        if (c == ' ') {
            c = '-';
        } else if (c == '\\' || c == '/' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
            c = '_';
        }
    }
    return s;
}

// Runs `exe_path` with `arg_string`, blocking until it exits, capturing the tail of its combined
// stdout+stderr into `log_out` (for error reporting on failure). Returns the exit code, or
// std::nullopt if the process could not even be started.
static std::optional<DWORD> runProcessSync(
    const std::string & exe_path, const std::string & arg_string, std::string & log_out
) noexcept {
    char temp_dir[MAX_PATH];
    char temp_path[MAX_PATH];
    if (GetTempPathA(MAX_PATH, temp_dir) == 0 ||
        GetTempFileNameA(temp_dir, "trt", 0, temp_path) == 0
    ) {
        return std::nullopt;
    }

    SECURITY_ATTRIBUTES sa{ sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE };
    HANDLE log_handle{ CreateFileA(
        temp_path, GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, nullptr
    ) };
    if (log_handle == INVALID_HANDLE_VALUE) {
        DeleteFileA(temp_path);
        return std::nullopt;
    }

    std::string cmdline{ "\"" + exe_path + "\" " + arg_string };
    std::vector<char> cmdline_buf(cmdline.begin(), cmdline.end());
    cmdline_buf.push_back('\0');

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = log_handle;
    si.hStdError = log_handle;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

    PROCESS_INFORMATION pi{};
    BOOL ok{ CreateProcessA(
        exe_path.c_str(), cmdline_buf.data(), nullptr, nullptr, /*bInheritHandles=*/TRUE,
        CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi
    ) };
    CloseHandle(log_handle);

    if (!ok) {
        DeleteFileA(temp_path);
        return std::nullopt;
    }

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code{ 1 };
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    {
        std::ifstream log_stream{ temp_path, std::ios::binary | std::ios::ate };
        if (log_stream.good()) {
            std::streamoff size{ log_stream.tellg() };
            constexpr std::streamoff max_tail{ 4000 };
            std::streamoff start{ size > max_tail ? size - max_tail : std::streamoff{ 0 } };
            log_stream.seekg(start, std::ios::beg);
            log_out.resize(static_cast<size_t>(size - start));
            log_stream.read(log_out.data(), static_cast<std::streamsize>(log_out.size()));
        }
    }
    DeleteFileA(temp_path);

    return exit_code;
}

// Resolves `requested_engine_path`/`onnx_path` down to a single, ready-to-load .engine file path,
// building one with trtexec.exe if necessary. Mirrors vsmlrt.py's own get_engine_path()/trtexec()
// build+cache logic -- needed because a static-shape TensorRT engine only accepts the exact shape
// it was built for, and unlike a deployment targeting one known resolution, arbitrary source
// material means the required shape isn't known until the clip is seen.
//
// Resolution rules:
//   - requested_engine_path given and already exists on disk: used as-is, no build.
//   - requested_engine_path given but missing, onnx_path given: built to exactly that path.
//   - requested_engine_path empty, onnx_path given: a path is derived (next to the onnx file,
//     encoding device name / TensorRT version / shape so distinct GPUs or TRT versions sharing a
//     cache directory can't collide) and used if it already exists, else built there.
//   - onnx_path empty and requested_engine_path missing or empty: error.
//
// Builds go to a per-process-unique temp filename first, then are moved into place, so a second
// process racing to build the exact same (new resolution, first-seen) engine concurrently can't
// corrupt either process's read of it -- whichever finishes first "wins" and the other discards
// its redundant copy.
static std::optional<std::string> resolveOrBuildEngine(
    const std::string & requested_engine_path, const std::string & onnx_path,
    const std::string & trtexec_path_param, int num_planes, int width, int height, int device_id,
    bool io_fp16, std::string & err_out
) noexcept {
    if (!requested_engine_path.empty()) {
        std::ifstream probe{ requested_engine_path, std::ios::binary };
        if (probe.good()) {
            return requested_engine_path;
        }
    }

    if (onnx_path.empty()) {
        err_out = requested_engine_path.empty() ?
            "either engine_path or onnx_path must be specified." :
            ("failed to open engine: " + requested_engine_path);
        return std::nullopt;
    }

    std::string final_engine_path{ requested_engine_path };
    if (final_engine_path.empty()) {
        cudaDeviceProp prop{};
        std::string device_name{
            cudaGetDeviceProperties(&prop, device_id) == cudaSuccess ? prop.name : ("device" + std::to_string(device_id))
        };
        device_name = sanitizeForFilename(device_name);

        int trt_version{ getInferLibVersion() };

        std::string dir;
        std::string stem{ onnx_path };
        if (auto slash_pos{ onnx_path.find_last_of("\\/") }; slash_pos != std::string::npos) {
            dir = onnx_path.substr(0, slash_pos + 1);
            stem = onnx_path.substr(slash_pos + 1);
        }
        if (auto dot_pos{ stem.find_last_of('.') }; dot_pos != std::string::npos) {
            stem = stem.substr(0, dot_pos);
        }

        final_engine_path = dir + stem + "_" + device_name + "_trt" + std::to_string(trt_version) +
            "_" + std::to_string(width) + "x" + std::to_string(height) + (io_fp16 ? "_fp16io.engine" : "_fp16.engine");

        std::ifstream probe{ final_engine_path, std::ios::binary };
        if (probe.good()) {
            return final_engine_path;
        }
    }

    std::string trtexec_path{ trtexec_path_param };
    if (trtexec_path.empty()) {
        auto own_dir{ getOwnModuleDirectory() };
        if (!own_dir.has_value()) {
            err_out = "failed to determine mlrt_trt.dll's own directory while locating trtexec.exe.";
            return std::nullopt;
        }
        trtexec_path = own_dir.value() + "\\trtexec.exe";
    }
    if (std::ifstream probe{ trtexec_path, std::ios::binary }; !probe.good()) {
        err_out = "trtexec.exe not found at " + trtexec_path + " (needed to build an engine from " +
            onnx_path + "; pass trtexec_path explicitly, or place trtexec.exe next to mlrt_trt.dll).";
        return std::nullopt;
    }

    std::string tmp_engine_path{
        final_engine_path + "." + std::to_string(GetCurrentProcessId()) + ".building"
    };

    std::ostringstream args;
    args << "--onnx=\"" << onnx_path << "\""
         << " --timingCacheFile=\"" << final_engine_path << ".cache\""
         << " --device=" << device_id
         << " --saveEngine=\"" << tmp_engine_path << "\""
         << " --shapes=input:1x" << num_planes << "x" << height << "x" << width
         << " --fp16"
         << " --tacticSources=-CUBLAS,-CUBLAS_LT,-CUDNN,+EDGE_MASK_CONVOLUTIONS,+JIT_CONVOLUTIONS"
         << " --noTF32"
         << (io_fp16 ? " --inputIOFormats=fp16:chw" : " --inputIOFormats=fp32:chw")
         << (io_fp16 ? " --outputIOFormats=fp16:chw" : " --outputIOFormats=fp32:chw")
         << " --builderOptimizationLevel=3";

    std::string log;
    auto exit_code{ runProcessSync(trtexec_path, args.str(), log) };
    if (!exit_code.has_value()) {
        err_out = "failed to launch trtexec.exe at " + trtexec_path;
        return std::nullopt;
    }
    if (exit_code.value() != 0) {
        DeleteFileA(tmp_engine_path.c_str());
        err_out = "trtexec.exe failed building an engine from " + onnx_path + " (exit code " +
            std::to_string(exit_code.value()) + "):\n" + log;
        return std::nullopt;
    }

    if (std::ifstream probe{ final_engine_path, std::ios::binary }; probe.good()) {
        // a concurrent process building this exact (new resolution, first-seen) engine already
        // finished first -- use theirs instead of racing to overwrite it.
        DeleteFileA(tmp_engine_path.c_str());
    } else if (!MoveFileExA(tmp_engine_path.c_str(), final_engine_path.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        err_out = "trtexec.exe succeeded but failed to move the built engine into place (" +
            tmp_engine_path + " -> " + final_engine_path + ").";
        return std::nullopt;
    }

    return final_engine_path;
}
#else
#define translateName(n) (n)
#endif

using namespace std::string_literals;

// only TensorRT >= 10 (this plugin is built against TensorRT 10.16 headers/libs only,
// no support for the pre-8.5/pre-10 binding-index APIs that vs_tensorrt.cpp still carries)

struct TicketSemaphore {
    std::atomic<intptr_t> ticket{};
    std::atomic<intptr_t> current{};

    void init(intptr_t num) noexcept {
        current.store(num, std::memory_order::seq_cst);
    }

    void acquire() noexcept {
        intptr_t tk{ ticket.fetch_add(1, std::memory_order::acquire) };
        while (true) {
            intptr_t curr{ current.load(std::memory_order::acquire) };
            if (tk < curr) {
                return;
            }
#if __cpp_lib_atomic_wait
            current.wait(curr, std::memory_order::relaxed);
#else
            std::this_thread::sleep_for(10ms);
#endif
        }
    }

    void release() noexcept {
        current.fetch_add(1, std::memory_order::release);
#if __cpp_lib_atomic_wait
        current.notify_all();
#endif
    }
};

struct TRTData {
    std::unique_ptr<AVS_Clip * []> nodes;
    int num_nodes;

    int overlap_w, overlap_h;
    int device_id;
    bool use_cuda_graph;
    int num_streams;

    // YUV mode: raw planar YUV in/out, RGB conversion + padding + crop on the GPU (rife_io.cu)
    bool yuv_mode{ false };
    bool cudadownload{ true };   // reserved for the device-frame (token) output of the mvtools_cuda interop
    RifeIoFormat yuv_fmt{};

    // mvtools_cuda device-frame interop (token frames in/out), only when the library with API v20 is present
    std::unique_ptr<MvCudaApi> mv;
    MvCudaFrameFormat mv_fmt{};
    unsigned long long producer{ 0 };
    AVS_Clip * child{ nullptr };   // frame0 clip (for regenerating evicted output frames)

    std::unique_ptr<nvinfer1::ILogger> logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;

    TicketSemaphore semaphore;
    std::vector<int> tickets;
    std::mutex ticket_lock;
    std::vector<InferenceInstance> instances;

    std::string err;
    bool timing_enabled;

    [[nodiscard]]
    int acquire() noexcept {
        semaphore.acquire();
        std::lock_guard<std::mutex> lock{ ticket_lock };
        int ticket{ tickets.back() };
        tickets.pop_back();
        return ticket;
    }

    void release(int ticket) noexcept {
        {
            std::lock_guard<std::mutex> lock{ ticket_lock };
            tickets.push_back(ticket);
        }
        semaphore.release();
    }
};

static AVS_VideoFrame * AVSC_CC get_frame_mlrt_trt(AVS_FilterInfo * fi, int n) {
    auto t0{ std::chrono::high_resolution_clock::now() };
    TRTData * d{ static_cast<TRTData *>(fi->user_data) };
    const int num_nodes{ d->num_nodes };

    std::unique_ptr<const AVS_VideoInfo * []> in_vis{ std::make_unique_for_overwrite<const AVS_VideoInfo * []>(num_nodes) };
    in_vis[0] = &fi->vi;

    std::unique_ptr<AVS_VideoFrame * []> src_frames{ std::make_unique_for_overwrite<AVS_VideoFrame * []>(num_nodes) };
    src_frames[0] = avs_get_frame(fi->child, n);
    if (!src_frames[0]) {
        return nullptr;
    }

    for (int i{ 1 }; i < num_nodes; ++i) {
        in_vis[i] = avs_get_video_info(d->nodes[i - 1]);
        src_frames[i] = avs_get_frame(d->nodes[i - 1], n);
        if (!src_frames[i]) {
            while (--i >= 0) {
                avs_release_video_frame(src_frames[i]);
            }
            return nullptr;
        }
    }

    const auto release_src = [&]() {
        for (int i{ 0 }; i < num_nodes; ++i) {
            avs_release_video_frame(src_frames[i]);
        }
    };

    auto t1{ std::chrono::high_resolution_clock::now() };
    const int ticket{ d->acquire() };
    auto t2{ std::chrono::high_resolution_clock::now() };
    InferenceInstance & instance{ d->instances[ticket] };

    auto input_name{ d->engine->getIOTensorName(0) };
    auto output_name{ d->engine->getIOTensorName(1) };
    const nvinfer1::Dims src_dim{ instance.exec_context->getTensorShape(input_name) };
    const nvinfer1::Dims dst_dim{ instance.exec_context->getTensorShape(output_name) };

    const int src_tile_h{ static_cast<int>(src_dim.d[2]) };
    const int src_tile_w{ static_cast<int>(src_dim.d[3]) };
    const int dst_tile_h{ static_cast<int>(dst_dim.d[2]) };
    const int dst_tile_w{ static_cast<int>(dst_dim.d[3]) };
    const int dst_planes{ static_cast<int>(dst_dim.d[1]) };

    constexpr int planes_r[3]{ AVS_PLANAR_R, AVS_PLANAR_G, AVS_PLANAR_B };
    constexpr int plane_y[1]{ AVS_PLANAR_Y };

    std::vector<const uint8_t *> src_ptrs;
    src_ptrs.reserve(src_dim.d[1]);
    for (int i{ 0 }; i < num_nodes; ++i) {
        const int * node_planes{ avs_is_rgb(in_vis[i]) ? planes_r : plane_y };
        for (int j{ 0 }; j < avs_num_components(in_vis[i]); ++j) {
            src_ptrs.emplace_back(avs_get_read_ptr_p(src_frames[i], node_planes[j]));
        }
    }

    AVS_VideoFrame * dst_frame{ avs_new_video_frame_p(fi->env, &fi->vi, src_frames[0]) };

    const int * dst_planes_arr{ avs_is_rgb(&fi->vi) ? planes_r : plane_y };
    std::vector<uint8_t *> dst_ptrs;
    dst_ptrs.reserve(dst_planes);
    for (int i{ 0 }; i < dst_planes; ++i) {
        dst_ptrs.emplace_back(avs_get_write_ptr_p(dst_frame, dst_planes_arr[i]));
    }

    const int node0_plane0{ avs_is_rgb(in_vis[0]) ? AVS_PLANAR_R : AVS_PLANAR_Y };
    const IOInfo info{
        .in = InputInfo{
            .width = avs_get_row_size_p(src_frames[0], node0_plane0) / avs_component_size(in_vis[0]),
            .height = avs_get_height_p(src_frames[0], node0_plane0),
            .pitch = avs_get_pitch_p(src_frames[0], node0_plane0),
            .bytes_per_sample = avs_component_size(in_vis[0]),
            .tile_w = src_tile_w,
            .tile_h = src_tile_h
        },
        .out = OutputInfo{
            .pitch = avs_get_pitch_p(dst_frame, dst_planes_arr[0]),
            .bytes_per_sample = avs_component_size(&fi->vi)
        },
        .w_scale = dst_tile_w / src_tile_w,
        .h_scale = dst_tile_h / src_tile_h,
        .overlap_w = d->overlap_w,
        .overlap_h = d->overlap_h
    };

    auto t3{ std::chrono::high_resolution_clock::now() };
    auto inference_result{ inference(
        fi->env, instance, d->device_id, d->use_cuda_graph, info, src_ptrs, dst_ptrs
    ) };
    auto t4{ std::chrono::high_resolution_clock::now() };

    d->release(ticket);
    release_src();

    if (d->timing_enabled) {
        auto us{ [](auto a, auto b) { return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count(); } };
        fprintf(stderr, "mlrt_trt frame %d: getframe=%lldus acquire=%lldus setup=%lldus inference=%lldus\n",
            n, (long long)us(t0, t1), (long long)us(t1, t2), (long long)us(t2, t3), (long long)us(t3, t4));
        fflush(stderr);
    }

    if (inference_result.has_value()) {
        d->err = "mlrt_trt: " + inference_result.value();
        fi->error = d->err.c_str();
        avs_release_video_frame(dst_frame);
        return nullptr;
    }

    return dst_frame;
}

static AVS_VideoFrame * AVSC_CC get_frame_mlrt_trt_yuv(AVS_FilterInfo * fi, int n) {
    auto t0{ std::chrono::high_resolution_clock::now() };
    TRTData * d{ static_cast<TRTData *>(fi->user_data) };

    AVS_VideoFrame * fa{ avs_get_frame(fi->child, n) };
    if (!fa) {
        return nullptr;
    }
    AVS_VideoFrame * fb{ avs_get_frame(d->nodes[0], n) };
    if (!fb) {
        avs_release_video_frame(fa);
        return nullptr;
    }
    AVS_VideoFrame * ft{ avs_get_frame(d->nodes[1], n) };
    if (!ft) {
        avs_release_video_frame(fa);
        avs_release_video_frame(fb);
        return nullptr;
    }
    const auto release_src{ [&]() {
        avs_release_video_frame(fa);
        avs_release_video_frame(fb);
        avs_release_video_frame(ft);
    } };

    // the timestep clip is a constant plane: only its first sample is used (never uploaded as a plane)
    const float timestep{ *reinterpret_cast<const float *>(avs_get_read_ptr_p(ft, AVS_PLANAR_Y)) };

    auto t1{ std::chrono::high_resolution_clock::now() };
    const int ticket{ d->acquire() };
    auto t2{ std::chrono::high_resolution_clock::now() };
    InferenceInstance & instance{ d->instances[ticket] };
    YuvResources & y{ instance.yuv };

    constexpr int planes[3]{ AVS_PLANAR_Y, AVS_PLANAR_U, AVS_PLANAR_V };

    const auto copy_in{ [&](AVS_VideoFrame * fr, uint8_t * dst) {
        for (int p{ 0 }; p < 3; ++p) {
            const int row_size{ avs_get_row_size_p(fr, planes[p]) };
            const int height{ avs_get_height_p(fr, planes[p]) };
            avs_bit_blt(fi->env, dst, row_size, avs_get_read_ptr_p(fr, planes[p]), avs_get_pitch_p(fr, planes[p]), row_size, height);
            dst += static_cast<size_t>(row_size) * height;
        }
    } };
    copy_in(fa, y.h_in[0].data);
    copy_in(fb, y.h_in[1].data);
    std::memcpy(y.h_ts.data, &timestep, sizeof(float));
    auto t3{ std::chrono::high_resolution_clock::now() };

    std::string error;
    if (cudaSetDevice(d->device_id) != cudaSuccess) {
        error = "cudaSetDevice failed";
    } else if (d->use_cuda_graph) {
        if (auto r{ cudaGraphLaunch(instance.graphexec, instance.stream) }; r != cudaSuccess) {
            error = std::string{ "cudaGraphLaunch failed: " } + cudaGetErrorString(r);
        }
    } else if (auto r{ enqueue(instance.src, instance.dst, instance.exec_context, instance.stream, &y) }; r.has_value()) {
        error = r.value();
    }
    if (error.empty()) {
        if (auto r{ cudaStreamSynchronize(instance.stream) }; r != cudaSuccess) {
            error = std::string{ "cudaStreamSynchronize failed: " } + cudaGetErrorString(r);
        }
    }
    auto t4{ std::chrono::high_resolution_clock::now() };

    if (error.empty() && y.timeline) {
        // GPU-side timestamps (ms) of the phase boundaries, relative to a process-wide base event
        static cudaEvent_t base{ [] { cudaEvent_t e; cudaEventCreate(&e); cudaEventRecord(e, nullptr); cudaEventSynchronize(e); return e; }() };
        float t[6]{};
        for (int k{ 0 }; k < 6; ++k) {
            cudaEventElapsedTime(&t[k], base, y.ev[k]);
        }
        static const auto host_base{ std::chrono::steady_clock::now() };
        const auto hms{ [&](auto tp) { return std::chrono::duration<double, std::milli>(tp - host_base).count(); } };
        // host times (ms): h_acq = ticket acquired (after getframe + acquire), h_sync = GPU work finished, h_in = getframe start
        fprintf(stderr, "TL inst=%d n=%d t0=%.3f t1=%.3f t2=%.3f t3=%.3f t4=%.3f t5=%.3f h_acq=%.3f h_sync=%.3f h_in=%.3f\n",
            ticket, n, t[0], t[1], t[2], t[3], t[4], t[5], hms(t2), hms(t4), hms(t0));
    }

    if (!error.empty()) {
        d->release(ticket);
        release_src();
        d->err = "mlrt_trt: " + error;
        fi->error = d->err.c_str();
        return nullptr;
    }

    AVS_VideoFrame * dst_frame{ avs_new_video_frame_p(fi->env, &fi->vi, fa) };
    {
        const uint8_t * src{ y.h_out.data };
        for (int p{ 0 }; p < 3; ++p) {
            const int row_size{ avs_get_row_size_p(dst_frame, planes[p]) };
            const int height{ avs_get_height_p(dst_frame, planes[p]) };
            avs_bit_blt(fi->env, avs_get_write_ptr_p(dst_frame, planes[p]), avs_get_pitch_p(dst_frame, planes[p]), src, row_size, row_size, height);
            src += static_cast<size_t>(row_size) * height;
        }
    }
    d->release(ticket);
    release_src();
    auto t5{ std::chrono::high_resolution_clock::now() };

    if (d->timing_enabled) {
        auto us{ [](auto a, auto b) { return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count(); } };
        fprintf(stderr, "mlrt_trt yuv frame %d: getframe=%lldus acquire=%lldus copyin=%lldus gpu=%lldus copyout=%lldus\n",
            n, (long long)us(t0, t1), (long long)us(t1, t2), (long long)us(t2, t3), (long long)us(t3, t4), (long long)us(t4, t5));
        fflush(stderr);
    }
    return dst_frame;
}

// ---------------------------------------------------------------------------------------------------------------------
// mvtools_cuda interop path (YUV mode). Inputs are token frames (device-resident) or ordinary frames (uploaded through the
// library's keyed upload, once per picture); the output is an ordinary frame (cudadownload=true) or a token frame.
// Written against the v20 interop API of MVTOOLS_CUDA_CHANGES.md.
// ---------------------------------------------------------------------------------------------------------------------

static bool mv_same_format(const MvCudaFrameFormat & a, const MvCudaFrameFormat & b) {
    return a.width == b.width && a.height == b.height && a.planes == b.planes && a.bytes_per_sample == b.bytes_per_sample &&
        a.bits_per_sample == b.bits_per_sample && a.x_ratio_uv == b.x_ratio_uv && a.y_ratio_uv == b.y_ratio_uv;
}

static std::string run_mv(
    TRTData * d, AVS_ScriptEnvironment * env, int n, AVS_VideoFrame * fa, AVS_VideoFrame * fb, float timestep,
    bool want_token, AVS_VideoFrame * dst_cpu, MvCudaTokenHeader * hdr_out
) {
    MvCudaApi & mv{ *d->mv };
    if (cudaSetDevice(d->device_id) != cudaSuccess) {
        return "cudaSetDevice failed";
    }

    constexpr int planes[3]{ AVS_PLANAR_Y, AVS_PLANAR_U, AVS_PLANAR_V };
    AVS_VideoFrame * const in_frames[2]{ fa, fb };
    const void * const identity[2]{ d->child, d->nodes[0] };

    // 1. device frames of the two input pictures
    MvCudaTokenHeader hdr[2]{};
    bool is_token[2]{};
    for (int i{ 0 }; i < 2; ++i) {
        is_token[i] = mv_read_token(in_frames[i], hdr[i]);
        if (is_token[i]) {
            if (!mv_same_format(hdr[i].format, d->mv_fmt)) {
                return "token frame format differs from the clip format";
            }
        } else {
            const void * src[3]{};
            int pitch[3]{};
            for (int p{ 0 }; p < 3; ++p) {
                src[p] = avs_get_read_ptr_p(in_frames[i], planes[p]);
                pitch[p] = avs_get_pitch_p(in_frames[i], planes[p]);
            }
            const unsigned long long key{
                (static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(identity[i])) * 0x9E3779B97F4A7C15ULL) ^
                (static_cast<unsigned long long>(n) + 0x1234567ULL)
            };
            if (mv.frame_upload_keyed(&d->mv_fmt, src, pitch, key, &hdr[i]) != 0) {
                return "mvcuda_frame_upload_keyed failed";
            }
        }
    }

    const int ticket{ d->acquire() };
    InferenceInstance & inst{ d->instances[ticket] };
    YuvResources & y{ inst.yuv };
    cudaStream_t stream{ inst.stream };
    const auto fail{ [&](const std::string & m) { d->release(ticket); return m; } };

    // 2. device pointers of the inputs (waits on their producers inside the library, never on the host)
    unsigned long long dev[2][3]{};
    int pitch[2][3]{};
    bool acquired[2]{};
    const auto release_inputs{ [&]() {
        for (int i{ 0 }; i < 2; ++i) {
            if (acquired[i]) {
                mv.token_release(hdr[i].token, stream);
                acquired[i] = false;
            }
        }
    } };
    for (int i{ 0 }; i < 2; ++i) {
        MvCudaFrameFormat f{};
        int rc{ mv.token_acquire(hdr[i].token, &f, dev[i], pitch[i], stream) };
        if (rc == MVCUDA_TOKEN_GONE && is_token[i] && hdr[i].producer) {
            // evicted from the registry: ask its producer to make the frame again
            MvCudaTokenHeader again{};
            if (mv.token_regenerate(hdr[i].producer, hdr[i].frame, env, &again) == 0) {
                hdr[i] = again;
                rc = mv.token_acquire(hdr[i].token, &f, dev[i], pitch[i], stream);
            }
        }
        if (rc != 0) {
            release_inputs();
            return fail("mvcuda_token_acquire failed (" + std::to_string(rc) + ")");
        }
        acquired[i] = true;
        if (pitch[i][1] != pitch[i][2]) {
            release_inputs();
            return fail("U and V planes of a device frame have different pitches (unsupported)");
        }
    }

    // 3. pack (YUV -> RGB fp16, padding) -> TensorRT -> unpack (RGB -> YUV, crop)
    const auto as_in{ [&](int i) {
        return RifeFrameIn{
            reinterpret_cast<const void *>(static_cast<uintptr_t>(dev[i][0])),
            reinterpret_cast<const void *>(static_cast<uintptr_t>(dev[i][1])),
            reinterpret_cast<const void *>(static_cast<uintptr_t>(dev[i][2])),
            pitch[i][0], pitch[i][1]
        };
    } };

    std::memcpy(y.h_ts.data, &timestep, sizeof(float));
    if (auto r{ cudaMemcpyAsync(y.d_ts.data, y.h_ts.data, sizeof(float), cudaMemcpyHostToDevice, stream) }; r != cudaSuccess) {
        release_inputs();
        return fail(std::string{ "timestep upload failed: " } + cudaGetErrorString(r));
    }
    if (auto r{ rife_pack(d->yuv_fmt, as_in(0), as_in(1), reinterpret_cast<const float *>(y.d_ts.data),
        reinterpret_cast<__half *>(inst.src.d_data.data), stream) }; r != cudaSuccess) {
        release_inputs();
        return fail(std::string{ "rife_pack failed: " } + cudaGetErrorString(r));
    }

    {
        const auto in_name{ d->engine->getIOTensorName(0) };
        const auto out_name{ d->engine->getIOTensorName(1) };
        if (!inst.exec_context->setTensorAddress(in_name, inst.src.d_data.data) ||
            !inst.exec_context->setTensorAddress(out_name, inst.dst.d_data.data) ||
            !inst.exec_context->enqueueV3(stream)) {
            release_inputs();
            return fail("TensorRT enqueue failed");
        }
    }

    MvCudaTokenHeader out_hdr{};
    if (want_token) {
        unsigned long long odev[3]{};
        int opitch[3]{};
        if (mv.frame_alloc(&d->mv_fmt, d->producer, n, odev, opitch, &out_hdr) != 0) {
            release_inputs();
            return fail("mvcuda_frame_alloc failed");
        }
        const RifeFrameOut out{
            reinterpret_cast<void *>(static_cast<uintptr_t>(odev[0])),
            reinterpret_cast<void *>(static_cast<uintptr_t>(odev[1])),
            reinterpret_cast<void *>(static_cast<uintptr_t>(odev[2])),
            opitch[0], opitch[1]
        };
        if (auto r{ rife_unpack(d->yuv_fmt, reinterpret_cast<const __half *>(inst.dst.d_data.data), out, stream) }; r != cudaSuccess) {
            mv.frame_discard(out_hdr.token);
            release_inputs();
            return fail(std::string{ "rife_unpack failed: " } + cudaGetErrorString(r));
        }
        if (mv.frame_commit(out_hdr.token, stream) != 0) {
            release_inputs();
            return fail("mvcuda_frame_commit failed");
        }
        release_inputs();
        // everything is ordered on the instance's stream: the next call using this instance queues behind it, so no host wait
        d->release(ticket);
        if (hdr_out) {
            *hdr_out = out_hdr;
        }
        return {};
    }

    // cudadownload: unpack into the instance's tight output buffer and bring only that (raw YUV) back
    if (auto r{ rife_unpack(d->yuv_fmt, reinterpret_cast<const __half *>(inst.dst.d_data.data), rife_tight_out(d->yuv_fmt, y.d_out.data), stream) };
        r != cudaSuccess) {
        release_inputs();
        return fail(std::string{ "rife_unpack failed: " } + cudaGetErrorString(r));
    }
    if (auto r{ cudaMemcpyAsync(y.h_out.data, y.d_out.data, y.frame_bytes, cudaMemcpyDeviceToHost, stream) }; r != cudaSuccess) {
        release_inputs();
        return fail(std::string{ "output download failed: " } + cudaGetErrorString(r));
    }
    release_inputs();
    if (auto r{ cudaStreamSynchronize(stream) }; r != cudaSuccess) {
        return fail(std::string{ "cudaStreamSynchronize failed: " } + cudaGetErrorString(r));
    }
    const uint8_t * src{ y.h_out.data };
    for (int p{ 0 }; p < 3; ++p) {
        const int row_size{ avs_get_row_size_p(dst_cpu, planes[p]) };
        const int height{ avs_get_height_p(dst_cpu, planes[p]) };
        avs_bit_blt(env, avs_get_write_ptr_p(dst_cpu, planes[p]), avs_get_pitch_p(dst_cpu, planes[p]), src, row_size, row_size, height);
        src += static_cast<size_t>(row_size) * height;
    }
    d->release(ticket);
    return {};
}

// producer callback of the library's regeneration mechanism: makes output frame `frame` again after an eviction
static int mv_regenerate(void * user, int frame, void * env, MvCudaTokenHeader * out) {
    TRTData * d{ static_cast<TRTData *>(user) };
    AVS_VideoFrame * fa{ avs_get_frame(d->child, frame) };
    if (!fa) {
        return -1;
    }
    AVS_VideoFrame * fb{ avs_get_frame(d->nodes[0], frame) };
    AVS_VideoFrame * ft{ fb ? avs_get_frame(d->nodes[1], frame) : nullptr };
    int rc{ -1 };
    if (fb && ft) {
        const float ts{ *reinterpret_cast<const float *>(avs_get_read_ptr_p(ft, AVS_PLANAR_Y)) };
        rc = run_mv(d, static_cast<AVS_ScriptEnvironment *>(env), frame, fa, fb, ts, true, nullptr, out).empty() ? 0 : -1;
    }
    avs_release_video_frame(fa);
    if (fb) avs_release_video_frame(fb);
    if (ft) avs_release_video_frame(ft);
    return rc;
}

static AVS_VideoFrame * AVSC_CC get_frame_mlrt_trt_mv(AVS_FilterInfo * fi, int n) {
    TRTData * d{ static_cast<TRTData *>(fi->user_data) };

    AVS_VideoFrame * fa{ avs_get_frame(fi->child, n) };
    if (!fa) {
        return nullptr;
    }
    AVS_VideoFrame * fb{ avs_get_frame(d->nodes[0], n) };
    AVS_VideoFrame * ft{ fb ? avs_get_frame(d->nodes[1], n) : nullptr };
    const auto release_src{ [&]() {
        avs_release_video_frame(fa);
        if (fb) avs_release_video_frame(fb);
        if (ft) avs_release_video_frame(ft);
    } };
    if (!fb || !ft) {
        release_src();
        return nullptr;
    }
    const float timestep{ *reinterpret_cast<const float *>(avs_get_read_ptr_p(ft, AVS_PLANAR_Y)) };

    const bool want_token{ !d->cudadownload };
    AVS_VideoFrame * dst{ avs_new_video_frame_p(fi->env, &fi->vi, fa) };
    MvCudaTokenHeader out_hdr{};
    const std::string error{ run_mv(d, fi->env, n, fa, fb, timestep, want_token, want_token ? nullptr : dst, &out_hdr) };
    release_src();
    if (!error.empty()) {
        avs_release_video_frame(dst);
        d->err = "mlrt_trt: " + error;
        fi->error = d->err.c_str();
        return nullptr;
    }
    if (want_token) {
        mv_write_token(dst, out_hdr);
    }
    return dst;
}

static void AVSC_CC free_mlrt_trt(AVS_FilterInfo * fi) {
    TRTData * d{ static_cast<TRTData *>(fi->user_data) };

    for (int i{ 0 }; i < d->num_nodes - 1; ++i) {
        avs_release_clip(d->nodes[i]);
    }

    cudaSetDevice(d->device_id);

    if (d->mv && d->producer) {
        d->mv->producer_unregister(d->producer);
    }

    delete d;
}

static int AVSC_CC set_cache_hints_mlrt_trt(AVS_FilterInfo * fi, int cachehints, int frame_range) {
    // MT_NICE_FILTER (1): ONE filter instance shared by all Prefetch threads. GetFrame is thread-safe (ticket semaphore +
    // per-stream buffers). MT_MULTI_INSTANCE (2) would give every Prefetch thread its own engine copy and its own
    // num_streams execution contexts: ~0.3 GB of GPU memory each, so Prefetch(12,..) or num_streams=3 overflowed an 8 GB
    // card into WDDM paging (10-100x slower).
    return cachehints == AVS_CACHE_GET_MTMODE ? 1 : 0;
}

static AVS_Value AVSC_CC Create_mlrt_trt(AVS_ScriptEnvironment * env, AVS_Value args, void * param) {
    enum {
        Clips, Engine_path, Overlap_w, Overlap_h, Tilesize_w, Tilesize_h,
        Device, Use_cuda_graph, Num_streams, Verbosity, Onnx_path, Trtexec_path,
        Cudadownload, Pad_multiple, Matrix, Range, Mvtools_cuda
    };

    // deliberately not a smart pointer: on error paths, avs_new_value_error() below returns a
    // pointer into d->err, which must stay alive after this function returns for AviSynth to
    // read it -- matching mlrt_ort.cpp's ORTData convention (leaked on error, freed by
    // free_mlrt_trt() on success).
    TRTData * d{ new TRTData() };
    d->timing_enabled = !!getenv("MLRT_TRT_TIMING");

    AVS_FilterInfo * fi;
    AVS_Clip * clip{ avs_new_c_filter(env, &fi, *avs_as_array(avs_array_elt(args, Clips)), 1) };

    const int num_nodes{ avs_array_size(avs_array_elt(args, Clips)) };
    d->num_nodes = num_nodes;
    if (num_nodes > 1) {
        d->nodes = std::make_unique_for_overwrite<AVS_Clip * []>(num_nodes - 1);
        for (int i{ 1 }; i < num_nodes; ++i) {
            d->nodes[i - 1] = avs_take_clip(*(avs_as_array(avs_array_elt(args, Clips)) + i), env);
        }
    }

    auto set_error{ [&](const std::string & error_message) {
        avs_release_clip(clip);
        for (int i{ 1 }; i < num_nodes; ++i) {
            avs_release_clip(d->nodes[i - 1]);
        }
        d->err = "mlrt_trt: " + error_message;
        return avs_new_value_error(d->err.c_str());
    } };

    if (avs_check_version(env, 10)) {
        return set_error("AviSynth+ version must be r3928 or later.");
    }

    // Pre-load cudart64_13.dll and nvinfer_10.dll from next to this DLL, before anything below
    // touches CUDA or TensorRT (cudart first: nvinfer_10.dll's own internals may call into it
    // too). See preloadFromOwnDirectory()'s comment for why this can't just rely on default
    // DLL search order -- both of these would otherwise resolve relative to whatever host
    // application (ffmpeg.exe, AvsPmod.exe, ...) happened to load this plugin.
    std::string preload_err;
    if (!preloadFromOwnDirectory("cudart64_13.dll", preload_err) ||
        !preloadFromOwnDirectory("nvinfer_10.dll", preload_err)
    ) {
        return set_error(preload_err);
    }

    std::unique_ptr<const AVS_VideoInfo * []> in_vis{ std::make_unique_for_overwrite<const AVS_VideoInfo * []>(num_nodes) };
    in_vis[0] = &fi->vi;
    for (int i{ 1 }; i < num_nodes; ++i) {
        in_vis[i] = avs_get_video_info(d->nodes[i - 1]);
    }
    // YUV mode: planar integer YUV (8..16 bit) clips [frame0, frame1, timestep(Y32)] -- the GPU does YUV<->RGB, padding
    // to the engine tile and the crop back; legacy mode: fp32 planar RGB / Y32 clips as before.
    const bool yuv_mode{
        avs_is_planar(in_vis[0]) && avs_is_yuv(in_vis[0]) && avs_num_components(in_vis[0]) == 3 &&
        avs_bits_per_component(in_vis[0]) >= 8 && avs_bits_per_component(in_vis[0]) <= 16
    };
    d->yuv_mode = yuv_mode;
    if (yuv_mode) {
        if (num_nodes != 3) {
            return set_error("YUV mode expects exactly 3 clips: frame0, frame1, timestep (Y32).");
        }
        if (in_vis[1]->pixel_type != in_vis[0]->pixel_type) {
            return set_error("frame0 and frame1 must have the same pixel type.");
        }
        if (avs_component_size(in_vis[2]) != 4 || avs_num_components(in_vis[2]) != 1) {
            return set_error("the timestep clip must be Y32.");
        }
    }
    for (int i{ 0 }; i < num_nodes; ++i) {
        if (!yuv_mode && avs_component_size(in_vis[i]) != 4) {
            return set_error("expects clip with type fp32.");
        }
        if (in_vis[i]->width != in_vis[0]->width || in_vis[i]->height != in_vis[0]->height) {
            return set_error("dimensions of clips mismatch.");
        }
        if (in_vis[i]->num_frames != in_vis[0]->num_frames) {
            return set_error("number of frames mismatch.");
        }
    }

    int error1, error2;
    d->overlap_w = avs_defined(avs_array_elt(args, Overlap_w)) ? avs_as_int(avs_array_elt(args, Overlap_w)) : 0;
    d->overlap_h = avs_defined(avs_array_elt(args, Overlap_h)) ? avs_as_int(avs_array_elt(args, Overlap_h)) : d->overlap_w;
    if (d->overlap_w < 0 || d->overlap_h < 0) {
        return set_error("overlap must be non-negative.");
    }

    int tile_w{ avs_defined(avs_array_elt(args, Tilesize_w)) ? avs_as_int(avs_array_elt(args, Tilesize_w)) : -4525 };
    int tile_h{ avs_defined(avs_array_elt(args, Tilesize_h)) ? avs_as_int(avs_array_elt(args, Tilesize_h)) : -4525 };
    TileSize tile_size;
    if (tile_w != -4525) {
        if (tile_h == -4525) {
            tile_h = tile_w;
        }
        if (tile_w - 2 * d->overlap_w <= 0 || tile_h - 2 * d->overlap_h <= 0) {
            return set_error("overlap too large.");
        }
        tile_size = RequestedTileSize{ .tile_w = tile_w, .tile_h = tile_h };
    } else {
        if (d->overlap_w != 0 || d->overlap_h != 0) {
            return set_error("tilesize must be specified.");
        }
        tile_size = VideoSize{ .width = in_vis[0]->width, .height = in_vis[0]->height };
    }

    if (yuv_mode) {
        if (d->overlap_w != 0 || d->overlap_h != 0 || tile_w != -4525) {
            return set_error("tiling (overlap/tilesize) is not supported in YUV mode.");
        }
        const int m{ avs_defined(avs_array_elt(args, Pad_multiple)) ? avs_as_int(avs_array_elt(args, Pad_multiple)) : 32 };
        if (m <= 0) {
            return set_error("pad_multiple must be positive.");
        }
        tile_size = VideoSize{
            .width = (in_vis[0]->width + m - 1) / m * m,
            .height = (in_vis[0]->height + m - 1) / m * m
        };

        d->cudadownload = avs_defined(avs_array_elt(args, Cudadownload)) ? !!avs_as_bool(avs_array_elt(args, Cudadownload)) : true;

        const std::string matrix{ avs_defined(avs_array_elt(args, Matrix)) ? avs_as_string(avs_array_elt(args, Matrix)) : "709" };
        const std::string range{ avs_defined(avs_array_elt(args, Range)) ? avs_as_string(avs_array_elt(args, Range)) : "limited" };
        float kr, kb;
        if (matrix == "709") { kr = 0.2126f; kb = 0.0722f; }
        else if (matrix == "601") { kr = 0.299f; kb = 0.114f; }
        else if (matrix == "2020") { kr = 0.2627f; kb = 0.0593f; }
        else { return set_error("matrix must be \"709\", \"601\" or \"2020\"."); }
        if (range != "limited" && range != "full") {
            return set_error("range must be \"limited\" or \"full\".");
        }
        const auto & v0{ *in_vis[0] };
        d->yuv_fmt = RifeIoFormat{
            .width = v0.width, .height = v0.height,
            .pad_w = std::get<VideoSize>(tile_size).width, .pad_h = std::get<VideoSize>(tile_size).height,
            .bits = avs_bits_per_component(&v0),
            .xr = 1 << avs_get_plane_width_subsampling(&v0, AVS_PLANAR_U),
            .yr = 1 << avs_get_plane_height_subsampling(&v0, AVS_PLANAR_U),
            .kr = kr, .kb = kb, .full_range = range == "full" ? 1 : 0
        };
    }

    int error;
    d->device_id = avs_defined(avs_array_elt(args, Device)) ? avs_as_int(avs_array_elt(args, Device)) : 0;
    int device_count;
    // must be set before the device context is created (before any other cuda*() call touching
    // this device) -- without it, cudaStreamSynchronize()'s default "auto" scheduling policy tends
    // to busy-spin while waiting for the GPU, and under AviSynth's Prefetch() with several worker
    // threads each blocked in a synchronize call concurrently, that spinning starves the CPU badly
    // enough to make concurrent num_streams>1 dramatically SLOWER than fully serialized (observed:
    // ~10x-250x slower per call under Prefetch(16,20)+num_streams=4 vs. num_streams=1).
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
        d->device_id < 0 || d->device_id >= device_count
    ) {
        return set_error("invalid device ID (" + std::to_string(d->device_id) + ").");
    }
    cudaSetDevice(d->device_id);

    d->use_cuda_graph = avs_defined(avs_array_elt(args, Use_cuda_graph)) ? !!avs_as_bool(avs_array_elt(args, Use_cuda_graph)) : false;
    d->num_streams = avs_defined(avs_array_elt(args, Num_streams)) ? avs_as_int(avs_array_elt(args, Num_streams)) : 1;
    if (d->num_streams <= 0) {
        return set_error("num_streams must be positive.");
    }

    int verbosity{ avs_defined(avs_array_elt(args, Verbosity)) ? avs_as_int(avs_array_elt(args, Verbosity)) : static_cast<int>(nvinfer1::ILogger::Severity::kWARNING) };
    d->logger = std::make_unique<Logger>();
    static_cast<Logger *>(d->logger.get())->set_verbosity(static_cast<nvinfer1::ILogger::Severity>(verbosity));

    std::string engine_path{ avs_defined(avs_array_elt(args, Engine_path)) ? avs_as_string(avs_array_elt(args, Engine_path)) : "" };
    std::string onnx_path{ avs_defined(avs_array_elt(args, Onnx_path)) ? avs_as_string(avs_array_elt(args, Onnx_path)) : "" };
    std::string trtexec_path{ avs_defined(avs_array_elt(args, Trtexec_path)) ? avs_as_string(avs_array_elt(args, Trtexec_path)) : "" };

    // resolveOrBuildEngine() needs the exact static shape the engine will be run at, which is
    // whatever tile_size already resolved to above (the full frame, unless tiling was requested).
    int build_width, build_height;
    std::visit([&](auto && ts) {
        using T = std::decay_t<decltype(ts)>;
        if constexpr (std::is_same_v<T, RequestedTileSize>) {
            build_width = ts.tile_w;
            build_height = ts.tile_h;
        } else {
            build_width = ts.width;
            build_height = ts.height;
        }
    }, tile_size);

    int num_planes{ 0 };
    for (int j{ 0 }; j < num_nodes; ++j) {
        num_planes += avs_num_components(in_vis[j]);
    }

#ifdef _WIN32
    {
        std::string resolve_err;
        auto resolved{ resolveOrBuildEngine(
            engine_path, onnx_path, trtexec_path, num_planes, build_width, build_height, d->device_id, yuv_mode, resolve_err
        ) };
        if (!resolved.has_value()) {
            return set_error(resolve_err);
        }
        engine_path = resolved.value();
    }
#else
    if (!onnx_path.empty()) {
        return set_error("onnx_path (on-demand engine building via trtexec) is only implemented on Windows.");
    }
    if (engine_path.empty()) {
        return set_error("engine_path must be specified.");
    }
#endif

    std::ifstream engine_stream{ translateName(engine_path.c_str()), std::ios::binary | std::ios::ate };
    if (!engine_stream.good()) {
        return set_error("failed to open engine: " + engine_path);
    }
    auto engine_nbytes{ engine_stream.tellg() };
    if (engine_nbytes == -1) {
        return set_error("failed to open engine: " + engine_path);
    }
    std::unique_ptr<char[], decltype(&free)> engine_data{
        static_cast<char *>(malloc(static_cast<size_t>(engine_nbytes))), free
    };
    engine_stream.seekg(0, std::ios::beg);
    engine_stream.read(engine_data.get(), static_cast<std::streamsize>(engine_nbytes));

    d->runtime.reset(nvinfer1::createInferRuntime(*d->logger));
    if (!d->runtime) {
        return set_error("failed to create TensorRT runtime.");
    }

    {
        auto maybe_engine{ initEngine(
            engine_data.get(), static_cast<size_t>(engine_nbytes), d->runtime, /*flexible_output=*/false
        ) };
        if (std::holds_alternative<std::unique_ptr<nvinfer1::ICudaEngine>>(maybe_engine)) {
            d->engine = std::move(std::get<std::unique_ptr<nvinfer1::ICudaEngine>>(maybe_engine));
        } else {
            return set_error(std::get<ErrorMessage>(maybe_engine));
        }
    }

    auto profile_index{ selectProfile(d->engine, tile_size) };

    bool is_dynamic{ false };
    {
        auto input_name{ d->engine->getIOTensorName(0) };
        auto input_shape{ d->engine->getTensorShape(input_name) };
        for (int32_t i{ 0 }; i < input_shape.nbDims; ++i) {
            if (input_shape.d[i] == -1) {
                is_dynamic = true;
                break;
            }
        }
    }

    d->instances.reserve(d->num_streams);
    for (int i{ 0 }; i < d->num_streams; ++i) {
        auto maybe_instance{ getInstance(d->engine, profile_index, tile_size, d->use_cuda_graph, is_dynamic, yuv_mode ? &d->yuv_fmt : nullptr) };
        if (std::holds_alternative<InferenceInstance>(maybe_instance)) {
            auto instance{ std::move(std::get<InferenceInstance>(maybe_instance)) };

            auto input_name{ instance.exec_context->getEngine().getIOTensorName(0) };
            const nvinfer1::Dims & network_in_dims{ instance.exec_context->getTensorShape(input_name) };
            if (network_in_dims.d[1] != num_planes) {
                return set_error("expects " + std::to_string(network_in_dims.d[1]) + " input planes.");
            }
            if (yuv_mode) {
                if (network_in_dims.d[2] < in_vis[0]->height || network_in_dims.d[3] < in_vis[0]->width) {
                    return set_error("engine input is smaller than the clip.");
                }
            } else if (network_in_dims.d[2] > in_vis[0]->height || network_in_dims.d[3] > in_vis[0]->width) {
                return set_error("tile size larger than clip dimension.");
            }

            d->instances.emplace_back(std::move(instance));
        } else {
            return set_error(std::get<ErrorMessage>(maybe_instance));
        }
    }

    d->semaphore.init(d->num_streams);
    d->tickets.reserve(d->num_streams);
    for (int i{ 0 }; i < d->num_streams; ++i) {
        d->tickets.push_back(i);
    }

    {
        auto input_name{ d->engine->getIOTensorName(0) };
        auto output_name{ d->engine->getIOTensorName(1) };

        const auto io_type{ yuv_mode ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT };
        if (d->engine->getTensorDataType(input_name) != io_type ||
            d->engine->getTensorDataType(output_name) != io_type
        ) {
            return set_error(yuv_mode ?
                "YUV mode needs an engine with fp16 input/output tensors (build with --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw, or pass onnx_path)." :
                "engine input/output tensors must be fp32 (use a YUV clip for fp16-I/O engines).");
        }

        const nvinfer1::Dims & in_dims{ d->instances[0].exec_context->getTensorShape(input_name) };
        const nvinfer1::Dims & out_dims{ d->instances[0].exec_context->getTensorShape(output_name) };

        if (yuv_mode) {
            if (out_dims.d[1] != 3 || out_dims.d[2] != in_dims.d[2] || out_dims.d[3] != in_dims.d[3]) {
                return set_error("YUV mode needs a 3-channel output of the same size as the input (RIFE).");
            }
            // output keeps the clip's pixel type and size (padding and crop happen on the GPU)
        } else {
        fi->vi.height *= static_cast<int>(out_dims.d[2]) / static_cast<int>(in_dims.d[2]);
        fi->vi.width *= static_cast<int>(out_dims.d[3]) / static_cast<int>(in_dims.d[3]);

        if (out_dims.d[1] == 1) {
            fi->vi.pixel_type = AVS_CS_Y32;
        } else if (out_dims.d[1] == 3) {
            fi->vi.pixel_type = AVS_CS_RGBPS;
        } else {
            return set_error("output must have 1 or 3 channels.");
        }
        }
    }

    // mvtools_cuda interop (token frames): only in YUV mode, only if the library with the v20 API is there
    if (yuv_mode && (!avs_defined(avs_array_elt(args, Mvtools_cuda)) || avs_as_bool(avs_array_elt(args, Mvtools_cuda)))) {
        auto own_dir{ getOwnModuleDirectory() };
        auto api{ MvCudaApi::load(own_dir.value_or("")) };
        if (api.ok) {
            const auto & v0{ *in_vis[0] };
            d->mv = std::make_unique<MvCudaApi>(api);
            d->mv_fmt = MvCudaFrameFormat{
                .width = v0.width, .height = v0.height, .planes = 3,
                .bytes_per_sample = avs_component_size(&v0), .bits_per_sample = avs_bits_per_component(&v0),
                .x_ratio_uv = d->yuv_fmt.xr, .y_ratio_uv = d->yuv_fmt.yr
            };
            d->child = fi->child;
            d->producer = d->mv->producer_register(mv_regenerate, d);
        } else if (verbosity >= static_cast<int>(nvinfer1::ILogger::Severity::kINFO)) {
            fprintf(stderr, "mlrt_trt: mvtools_cuda interop off (%s); using the standalone YUV path.\n", api.why_not.c_str());
        }
    }

    AVS_Value v{ avs_new_value_clip(clip) };

    fi->user_data = static_cast<void *>(d);
    fi->get_frame = yuv_mode ? (d->mv ? get_frame_mlrt_trt_mv : get_frame_mlrt_trt_yuv) : get_frame_mlrt_trt;
    fi->free_filter = free_mlrt_trt;
    fi->set_cache_hints = set_cache_hints_mlrt_trt;

    avs_release_clip(clip);

    return v;
}

const char * AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment * env) {
    avs_add_function(
        env, "mlrt_trt",
        "c+[engine_path]s[overlap_w]i[overlap_h]i[tilesize_w]i[tilesize_h]i"
        "[device]i[use_cuda_graph]b[num_streams]i[verbosity]i[onnx_path]s[trtexec_path]s"
        "[cudadownload]b[pad_multiple]i[matrix]s[range]s[mvtools_cuda]b",
        Create_mlrt_trt, 0
    );
    return "mlrt_trt";
}

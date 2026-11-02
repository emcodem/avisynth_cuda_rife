// Run-time interface to mvtools_cuda.dll (github.com/emcodem/mvtools, branch gpu_work): device-resident "token frames".
// The struct layouts and constants below mirror Sources/cuda/mvcuda.h (API v19) -- they are the ABI of the library.
// The five interop functions (token_acquire / token_release / frame_alloc / frame_commit / frame_discard) are the
// v20 additions specified in MVTOOLS_CUDA_CHANGES.md. Everything is resolved with GetProcAddress, so this plugin runs
// standalone when the library is missing, has no device, or reports another API version.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>

#ifdef _WIN32
#include <Windows.h>
#endif

#include <avisynth_c.h>

struct MvCudaFrameFormat {
    int width, height;          // luma size in samples
    int planes;                 // 1 (Y only) or 3 (Y, U, V)
    int bytes_per_sample;       // 1 or 2
    int bits_per_sample;        // 8..16
    int x_ratio_uv, y_ratio_uv; // chroma subsampling (1 or 2)
};

struct MvCudaTokenHeader {
    unsigned long long magic;    // MVCUDA_TOKEN_MAGIC
    int layout;                  // MVCUDA_TOKEN_LAYOUT
    int device;                  // CUDA device index the frame lives on
    unsigned long long token;    // registry key
    unsigned long long producer; // mvcuda_producer_register id of the filter that made it (0: none)
    int frame;                   // frame number at the producer
    int reserved;
    MvCudaFrameFormat format;
};

constexpr unsigned long long MVCUDA_TOKEN_MAGIC = 0x4B5441445543564DULL; // "MVCUDATK"
constexpr int MVCUDA_TOKEN_LAYOUT = 1;
constexpr int MVCUDA_TOKEN_GONE = -6;
constexpr int MVCUDA_API_VERSION_NEEDED = 20;

using MvCudaRegenFn = int (*)(void * user, int frame, void * env, MvCudaTokenHeader * out);

struct MvCudaApi {
    bool ok{ false };
    std::string why_not;   // reason when !ok (diagnostics only)

    int (*api_version)() {};
    int (*device_count)() {};
    int (*frame_upload_keyed)(const MvCudaFrameFormat *, const void * const[3], const int[3], unsigned long long, MvCudaTokenHeader *) {};
    int (*frame_download)(unsigned long long, void * const[3], const int[3]) {};
    unsigned long long (*producer_register)(MvCudaRegenFn, void *) {};
    void (*producer_unregister)(unsigned long long) {};
    int (*token_regenerate)(unsigned long long, int, void *, MvCudaTokenHeader *) {};
    int (*token_pin)(unsigned long long) {};
    void (*token_unpin)(unsigned long long) {};
    // v20
    int (*token_acquire)(unsigned long long, MvCudaFrameFormat *, unsigned long long[3], int[3], void *) {};
    void (*token_release)(unsigned long long, void *) {};
    int (*frame_alloc)(const MvCudaFrameFormat *, unsigned long long, int, unsigned long long[3], int[3], MvCudaTokenHeader *) {};
    int (*frame_commit)(unsigned long long, void *) {};
    void (*frame_discard)(unsigned long long) {};

    // Loads (or finds the already loaded) mvtools_cuda.dll. If another plugin (mvtools2 / cudafilters) already loaded it,
    // that same instance is used -- the frame registry is per library instance, so token frames only work when every
    // plugin shares one. Otherwise it is loaded from `own_dir` (next to this plugin).
    static MvCudaApi load(const std::string & own_dir) {
        MvCudaApi a;
#ifdef _WIN32
        HMODULE h = GetModuleHandleA("mvtools_cuda.dll");
        if (!h && !own_dir.empty()) {
            const std::string full = own_dir + "\\mvtools_cuda.dll";
            h = LoadLibraryExA(full.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        }
        if (!h) {
            a.why_not = "mvtools_cuda.dll not found";
            return a;
        }
        const auto sym = [&](const char * name) { return reinterpret_cast<void *>(GetProcAddress(h, name)); };
#define MVCUDA_BIND(field, name) a.field = reinterpret_cast<decltype(a.field)>(sym(name)); if (!a.field) { a.why_not = std::string("missing symbol ") + name; return a; }
        MVCUDA_BIND(api_version, "mvcuda_api_version")
        MVCUDA_BIND(device_count, "mvcuda_device_count")
        if (a.api_version() != MVCUDA_API_VERSION_NEEDED) {
            a.why_not = "mvtools_cuda API version " + std::to_string(a.api_version()) + ", need " + std::to_string(MVCUDA_API_VERSION_NEEDED);
            return a;
        }
        if (a.device_count() <= 0) {
            a.why_not = "no CUDA device for mvtools_cuda";
            return a;
        }
        MVCUDA_BIND(frame_upload_keyed, "mvcuda_frame_upload_keyed")
        MVCUDA_BIND(frame_download, "mvcuda_frame_download")
        MVCUDA_BIND(producer_register, "mvcuda_producer_register")
        MVCUDA_BIND(producer_unregister, "mvcuda_producer_unregister")
        MVCUDA_BIND(token_regenerate, "mvcuda_token_regenerate")
        MVCUDA_BIND(token_pin, "mvcuda_token_pin")
        MVCUDA_BIND(token_unpin, "mvcuda_token_unpin")
        MVCUDA_BIND(token_acquire, "mvcuda_token_acquire")
        MVCUDA_BIND(token_release, "mvcuda_token_release")
        MVCUDA_BIND(frame_alloc, "mvcuda_frame_alloc")
        MVCUDA_BIND(frame_commit, "mvcuda_frame_commit")
        MVCUDA_BIND(frame_discard, "mvcuda_frame_discard")
#undef MVCUDA_BIND
        a.ok = true;
#else
        a.why_not = "mvtools_cuda interop is only implemented on Windows";
#endif
        return a;
    }
};

// ---- token frame helpers (same convention as cudafilters/CudaCommon.h) ----

// A token frame is an ordinary AviSynth frame whose first luma row holds the header.
static inline bool mv_read_token(AVS_VideoFrame * f, MvCudaTokenHeader & h) {
    if (avs_get_row_size_p(f, AVS_PLANAR_Y) < static_cast<int>(sizeof(MvCudaTokenHeader))) {
        return false;
    }
    std::memcpy(&h, avs_get_read_ptr_p(f, AVS_PLANAR_Y), sizeof(h));
    return h.magic == MVCUDA_TOKEN_MAGIC && h.layout == MVCUDA_TOKEN_LAYOUT;
}

// Header into luma row 0, every 8th luma row black (a CPU filter shows stripes), chroma neutral.
static inline void mv_write_token(AVS_VideoFrame * dst, const MvCudaTokenHeader & h) {
    const int bits = h.format.bits_per_sample;
    const bool wide = h.format.bytes_per_sample == 2;
    const int black = 16 << (bits - 8), mid = 128 << (bits - 8);
    const int planes[3] = { AVS_PLANAR_Y, AVS_PLANAR_U, AVS_PLANAR_V };
    for (int p = 0; p < h.format.planes; ++p) {
        uint8_t * base = avs_get_write_ptr_p(dst, planes[p]);
        const int pitch = avs_get_pitch_p(dst, planes[p]);
        const int rows = avs_get_height_p(dst, planes[p]);
        const int row_size = avs_get_row_size_p(dst, planes[p]);
        const int value = p == 0 ? black : mid;
        for (int y = 0; y < rows; ++y) {
            uint8_t * r = base + static_cast<size_t>(y) * pitch;
            if (wide) {
                uint16_t * w = reinterpret_cast<uint16_t *>(r);
                const uint16_t v = static_cast<uint16_t>(value);
                const bool stripe = p == 0 && (y % 8) == 0;
                for (int x = 0; x < row_size / 2; ++x) w[x] = stripe ? static_cast<uint16_t>(black) : v;
            } else {
                std::memset(r, p == 0 && (y % 8) == 0 ? black : value, static_cast<size_t>(row_size));
            }
        }
    }
    std::memcpy(avs_get_write_ptr_p(dst, AVS_PLANAR_Y), &h, sizeof(h));
}

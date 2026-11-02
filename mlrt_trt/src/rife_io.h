// GPU-side I/O stages of the RIFE pipeline: raw planar YUV (8..16 bit integer, 4:2:0 / 4:2:2 / 4:4:4) <-> the fp16 RGB
// tensors of an fp16-I/O TensorRT engine. Replaces the CPU-side zimg conversions + AddBorders/Crop of the script, and moves
// only raw YUV over PCIe instead of fp32 RGB.
//
// Matches z_ConvertFormat("709:709:709:l=>rgb:709:709:f") / the reverse (verified numerically, see tools/):
//   * chroma 4:2:0 / 4:2:2: left co-sited horizontally, vertically centred (MPEG-2 / H.264 default siting)
//   * chroma upsampling: Catmull-Rom (bicubic b=0 c=0.5), edge replicate
//   * chroma downsampling: Catmull-Rom stretched by the ratio, edge replicate; round-to-nearest quantisation
//   * RGB is not clamped (like zimg's float RGB)
#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct RifeIoFormat {
    int width, height;      // luma size of the clip (what the user sees)
    int pad_w, pad_h;       // engine tile size (>= width/height); area beyond the clip is black (RGB 0), like AddBorders
    int bits;               // 8..16 (8: uint8 samples, otherwise uint16)
    int xr, yr;             // chroma subsampling (1 or 2)
    float kr, kb;           // matrix luma coefficients (BT.709: 0.2126, 0.0722)
    int full_range;         // 0 limited (16..235 / 16..240 scaled), 1 full
};

// Planes may be pitched (row pitch in BYTES, >= plane width * bytes per sample): tight host-staged frames pass
// width*bps / chroma_w*bps, device frames of the mvtools_cuda registry pass their allocation pitch.
// chroma plane size = ceil(width/xr) x ceil(height/yr)
struct RifeFrameIn  { const void * y; const void * u; const void * v; int y_pitch; int c_pitch; };
struct RifeFrameOut { void * y; void * u; void * v; int y_pitch; int c_pitch; };

// out: fp16 [7][pad_h][pad_w] = RGB of frame 0, RGB of frame 1, timestep plane. *d_ts: device float (the timestep).
cudaError_t rife_pack(const RifeIoFormat & f, const RifeFrameIn & f0, const RifeFrameIn & f1,
    const float * d_ts, __half * out, cudaStream_t stream);

// in: fp16 [3][pad_h][pad_w] (the engine output); y/u/v: tightly packed output planes of the clip size.
cudaError_t rife_unpack(const RifeIoFormat & f, const __half * in, const RifeFrameOut & out, cudaStream_t stream);

// host helpers
static inline int rife_chroma_w(const RifeIoFormat & f) { return (f.width + f.xr - 1) / f.xr; }
static inline int rife_chroma_h(const RifeIoFormat & f) { return (f.height + f.yr - 1) / f.yr; }
static inline size_t rife_bps(const RifeIoFormat & f) { return f.bits > 8 ? 2 : 1; }

// tight (unpitched) views of a frame stored as Y, U, V planes back to back at `base`
static inline RifeFrameIn rife_tight_in(const RifeIoFormat & f, const void * base) {
    const size_t ysz = static_cast<size_t>(f.width) * f.height * rife_bps(f);
    const size_t csz = static_cast<size_t>(rife_chroma_w(f)) * rife_chroma_h(f) * rife_bps(f);
    const char * b = static_cast<const char *>(base);
    return RifeFrameIn{ b, b + ysz, b + ysz + csz,
        static_cast<int>(f.width * rife_bps(f)), static_cast<int>(rife_chroma_w(f) * rife_bps(f)) };
}
static inline RifeFrameOut rife_tight_out(const RifeIoFormat & f, void * base) {
    const size_t ysz = static_cast<size_t>(f.width) * f.height * rife_bps(f);
    const size_t csz = static_cast<size_t>(rife_chroma_w(f)) * rife_chroma_h(f) * rife_bps(f);
    char * b = static_cast<char *>(base);
    return RifeFrameOut{ b, b + ysz, b + ysz + csz,
        static_cast<int>(f.width * rife_bps(f)), static_cast<int>(rife_chroma_w(f) * rife_bps(f)) };
}

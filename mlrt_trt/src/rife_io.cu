#include "rife_io.h"

namespace {

__device__ __forceinline__ float cr_w(float x) {
    x = fabsf(x);
    if (x < 1.f) return (1.5f * x - 2.5f) * x * x + 1.f;
    if (x < 2.f) return ((-0.5f * x + 2.5f) * x - 4.f) * x + 2.f;
    return 0.f;
}

// half-sample mirror at the borders (-1 -> 0, n -> n-1), what zimg does; then clamp for tiny planes
__device__ __forceinline__ int mirror(int i, int n) {
    if (i < 0) i = -i - 1;
    if (i >= n) i = 2 * n - 1 - i;
    return min(max(i, 0), n - 1);
}

template <typename T>
__device__ __forceinline__ const T * row(const T * p, int pitch_bytes, int y) {
    return reinterpret_cast<const T *>(reinterpret_cast<const char *>(p) + static_cast<size_t>(y) * pitch_bytes);
}

template <typename T>
__device__ __forceinline__ float ld(const T * p, int pitch, int w, int h, int x, int y) {
    x = mirror(x, w);
    y = mirror(y, h);
    return static_cast<float>(row(p, pitch, y)[x]);
}

// chroma plane sample interpolated at luma position (x, y)
template <typename T>
__device__ float chroma_up(const T * p, int pitch, int cw, int ch, int x, int y, int xr, int yr) {
    const float cx = xr == 2 ? x * 0.5f : static_cast<float>(x);                 // left co-sited
    const float cy = yr == 2 ? (y + 0.5f) * 0.5f - 0.5f : static_cast<float>(y); // vertically centred
    const int ix = static_cast<int>(floorf(cx));
    const int iy = static_cast<int>(floorf(cy));
    const float fx = cx - ix;
    const float fy = cy - iy;
    if (fx == 0.f && fy == 0.f) return ld(p, pitch, cw, ch, ix, iy);

    float wx[4] = { cr_w(fx + 1.f), cr_w(fx), cr_w(1.f - fx), cr_w(2.f - fx) };
    float wy[4] = { cr_w(fy + 1.f), cr_w(fy), cr_w(1.f - fy), cr_w(2.f - fy) };
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (wy[j] == 0.f) continue;
        float row_acc = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (wx[i] == 0.f) continue;
            row_acc += wx[i] * ld(p, pitch, cw, ch, ix - 1 + i, iy - 1 + j);
        }
        acc += wy[j] * row_acc;
    }
    return acc;
}

struct Coef {
    float yoff, yscale, coff, cscale;
    float inv_yscale, inv_cscale;
    float rcr, bcb, gcb, gcr;   // R = y + rcr*cr, B = y + bcb*cb, G = y - gcb*cb - gcr*cr
    float inv_rcr, inv_bcb;
    float kr, kg, kb;
    float maxv;
};

__host__ __device__ __forceinline__ Coef make_coef(const RifeIoFormat & f) {
    Coef c;
    const float s = static_cast<float>(1 << (f.bits - 8));
    c.maxv = static_cast<float>((1 << f.bits) - 1);
    if (f.full_range) {
        c.yoff = 0.f; c.yscale = c.maxv; c.coff = 128.f * s; c.cscale = c.maxv;
    } else {
        c.yoff = 16.f * s; c.yscale = 219.f * s; c.coff = 128.f * s; c.cscale = 224.f * s;
    }
    c.inv_yscale = 1.f / c.yscale; c.inv_cscale = 1.f / c.cscale;
    c.kr = f.kr; c.kb = f.kb; c.kg = 1.f - f.kr - f.kb;
    c.rcr = 2.f * (1.f - f.kr);
    c.bcb = 2.f * (1.f - f.kb);
    c.inv_rcr = 1.f / c.rcr; c.inv_bcb = 1.f / c.bcb;
    c.gcb = 2.f * f.kb * (1.f - f.kb) / c.kg;
    c.gcr = 2.f * f.kr * (1.f - f.kr) / c.kg;
    return c;
}

template <typename T>
__device__ __forceinline__ void yuv_to_rgb(const Coef & c, const RifeIoFormat & f, const T * yp, const T * up, const T * vp,
    int y_pitch, int c_pitch, int x, int y, float & r, float & g, float & b) {
    const int cw = (f.width + f.xr - 1) / f.xr;
    const int ch = (f.height + f.yr - 1) / f.yr;
    const float Y = static_cast<float>(row(yp, y_pitch, y)[x]);
    const float U = chroma_up(up, c_pitch, cw, ch, x, y, f.xr, f.yr);
    const float V = chroma_up(vp, c_pitch, cw, ch, x, y, f.xr, f.yr);
    const float yy = (Y - c.yoff) * c.inv_yscale;
    const float cb = (U - c.coff) * c.inv_cscale;
    const float cr = (V - c.coff) * c.inv_cscale;
    r = yy + c.rcr * cr;
    b = yy + c.bcb * cb;
    g = yy - c.gcb * cb - c.gcr * cr;
}

template <typename T>
__global__ void k_pack(RifeIoFormat f, const T * y0, const T * u0, const T * v0, const T * y1, const T * u1, const T * v1,
    int y_pitch, int c_pitch, const float * ts, __half * out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= f.pad_w || y >= f.pad_h) return;

    const size_t plane = static_cast<size_t>(f.pad_h) * f.pad_w;
    const size_t o = static_cast<size_t>(y) * f.pad_w + x;

    float r0 = 0.f, g0 = 0.f, b0 = 0.f, r1 = 0.f, g1 = 0.f, b1 = 0.f;
    if (x < f.width && y < f.height) {
        const Coef c = make_coef(f);
        yuv_to_rgb(c, f, y0, u0, v0, y_pitch, c_pitch, x, y, r0, g0, b0);
        yuv_to_rgb(c, f, y1, u1, v1, y_pitch, c_pitch, x, y, r1, g1, b1);
    }
    out[0 * plane + o] = __float2half(r0);
    out[1 * plane + o] = __float2half(g0);
    out[2 * plane + o] = __float2half(b0);
    out[3 * plane + o] = __float2half(r1);
    out[4 * plane + o] = __float2half(g1);
    out[5 * plane + o] = __float2half(b1);
    out[6 * plane + o] = __float2half(*ts);
}

__device__ __forceinline__ float rd(const __half * in, size_t plane, int pad_w, int c, int x, int y) {
    return __half2float(in[c * plane + static_cast<size_t>(y) * pad_w + x]);
}

template <typename T>
__device__ __forceinline__ T quant(float v, float maxv) {
    return static_cast<T>(fminf(fmaxf(floorf(v + 0.5f), 0.f), maxv));
}

template <typename T>
__global__ void k_unpack_y(RifeIoFormat f, const __half * in, T * yp, int y_pitch) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= f.width || y >= f.height) return;
    const Coef c = make_coef(f);
    const size_t plane = static_cast<size_t>(f.pad_h) * f.pad_w;
    const float r = rd(in, plane, f.pad_w, 0, x, y);
    const float g = rd(in, plane, f.pad_w, 1, x, y);
    const float b = rd(in, plane, f.pad_w, 2, x, y);
    const float Y = c.kr * r + c.kg * g + c.kb * b;
    reinterpret_cast<T *>(reinterpret_cast<char *>(yp) + static_cast<size_t>(y) * y_pitch)[x] = quant<T>(c.yoff + c.yscale * Y, c.maxv);
}

// chroma at luma position (lx, ly) of the full-resolution Cb / Cr computed from the RGB tensor
__device__ __forceinline__ void cbcr_at(const Coef & c, const RifeIoFormat & f, const __half * in, size_t plane, int lx, int ly,
    float & cb, float & cr) {
    lx = mirror(lx, f.width);
    ly = mirror(ly, f.height);
    const float r = rd(in, plane, f.pad_w, 0, lx, ly);
    const float g = rd(in, plane, f.pad_w, 1, lx, ly);
    const float b = rd(in, plane, f.pad_w, 2, lx, ly);
    const float Y = c.kr * r + c.kg * g + c.kb * b;
    cb = (b - Y) * c.inv_bcb;
    cr = (r - Y) * c.inv_rcr;
}

template <typename T>
__global__ void k_unpack_c(RifeIoFormat f, const __half * in, T * up, T * vp, int c_pitch) {
    const int cw = (f.width + f.xr - 1) / f.xr;
    const int ch = (f.height + f.yr - 1) / f.yr;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= cw || j >= ch) return;

    const Coef c = make_coef(f);
    const size_t plane = static_cast<size_t>(f.pad_h) * f.pad_w;

    // centre of this chroma sample in luma coordinates: left co-sited horizontally, vertically centred
    const float cx = f.xr == 2 ? 2.f * i : static_cast<float>(i);
    const float cy = f.yr == 2 ? 2.f * j + 0.5f : static_cast<float>(j);
    const float sx = static_cast<float>(f.xr);   // filter stretch = decimation ratio
    const float sy = static_cast<float>(f.yr);

    const int x0 = static_cast<int>(floorf(cx - 2.f * sx)) + 1, x1 = static_cast<int>(floorf(cx + 2.f * sx));
    const int y0 = static_cast<int>(floorf(cy - 2.f * sy)) + 1, y1 = static_cast<int>(floorf(cy + 2.f * sy));

    float sw = 0.f, scb = 0.f, scr = 0.f;
    for (int ly = y0; ly <= y1; ++ly) {
        const float wy = cr_w((ly - cy) / sy);
        if (wy == 0.f) continue;
        for (int lx = x0; lx <= x1; ++lx) {
            const float w = wy * cr_w((lx - cx) / sx);
            if (w == 0.f) continue;
            float cb, cr;
            cbcr_at(c, f, in, plane, lx, ly, cb, cr);
            sw += w; scb += w * cb; scr += w * cr;
        }
    }
    const float inv = 1.f / sw;
    const size_t ob = static_cast<size_t>(j) * c_pitch;
    reinterpret_cast<T *>(reinterpret_cast<char *>(up) + ob)[i] = quant<T>(c.coff + c.cscale * (scb * inv), c.maxv);
    reinterpret_cast<T *>(reinterpret_cast<char *>(vp) + ob)[i] = quant<T>(c.coff + c.cscale * (scr * inv), c.maxv);
}

// ---- fast pack for 4:2:0 / 4:2:2 (horizontal ratio 2) ------------------------------------------------------------------
// One thread handles a 2 x YR block of luma pixels (YR = 2: 4:2:0, YR = 1: 4:2:2) that share one chroma sample position, so
// the chroma window is loaded once and the Catmull-Rom weights are compile-time constants:
//   horizontal (left co-sited): even x = c[i],  odd x = (-c[i-1] + 9 c[i] + 9 c[i+1] - c[i+2]) / 16
//   vertical (centred, 4:2:0):  y = 2j   -> rows j-2..j+1 with (-3/128, 29/128, 111/128, -9/128)
//                               y = 2j+1 -> rows j-1..j+2 with (-9/128, 111/128, 29/128, -3/128)
template <typename T, int YR>
__device__ __forceinline__ void cell_rgb(const RifeIoFormat & f, const Coef & c, const T * yp, const T * up, const T * vp,
    int y_pitch, int c_pitch, int cx, int cy, float (&rgb)[YR][2][3]) {
    constexpr int NR = YR == 2 ? 5 : 1;
    const int cw = (f.width + 1) / 2;
    const int ch = (f.height + YR - 1) / YR;
    const int i0 = mirror(cx - 1, cw), i1 = mirror(cx, cw), i2 = mirror(cx + 1, cw), i3 = mirror(cx + 2, cw);

    float uh[NR][2], vh[NR][2];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const int crow = mirror(YR == 2 ? cy - 2 + r : cy, ch);
        const T * ur = row(up, c_pitch, crow);
        const T * vr = row(vp, c_pitch, crow);
        const float u1 = static_cast<float>(ur[i1]), v1 = static_cast<float>(vr[i1]);
        uh[r][0] = u1;
        vh[r][0] = v1;
        uh[r][1] = (9.f * (u1 + static_cast<float>(ur[i2])) - (static_cast<float>(ur[i0]) + static_cast<float>(ur[i3]))) * (1.f / 16.f);
        vh[r][1] = (9.f * (v1 + static_cast<float>(vr[i2])) - (static_cast<float>(vr[i0]) + static_cast<float>(vr[i3]))) * (1.f / 16.f);
    }

#pragma unroll
    for (int dy = 0; dy < YR; ++dy) {
        const int y = YR * cy + dy;
#pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            const int x = 2 * cx + dx;
            if (x >= f.width || y >= f.height) {
                rgb[dy][dx][0] = rgb[dy][dx][1] = rgb[dy][dx][2] = 0.f;
                continue;
            }
            float U, V;
            if (YR == 2) {
                const int r0 = dy;   // first window row of this output row
                const float w0 = dy == 0 ? -3.f / 128.f : -9.f / 128.f;
                const float w1 = dy == 0 ? 29.f / 128.f : 111.f / 128.f;
                const float w2 = dy == 0 ? 111.f / 128.f : 29.f / 128.f;
                const float w3 = dy == 0 ? -9.f / 128.f : -3.f / 128.f;
                U = w0 * uh[r0][dx] + w1 * uh[r0 + 1][dx] + w2 * uh[r0 + 2][dx] + w3 * uh[r0 + 3][dx];
                V = w0 * vh[r0][dx] + w1 * vh[r0 + 1][dx] + w2 * vh[r0 + 2][dx] + w3 * vh[r0 + 3][dx];
            } else {
                U = uh[0][dx];
                V = vh[0][dx];
            }
            const float Y = static_cast<float>(row(yp, y_pitch, y)[x]);
            const float yy = (Y - c.yoff) * c.inv_yscale;
            const float cb = (U - c.coff) * c.inv_cscale;
            const float cr = (V - c.coff) * c.inv_cscale;
            rgb[dy][dx][0] = yy + c.rcr * cr;
            rgb[dy][dx][2] = yy + c.bcb * cb;
            rgb[dy][dx][1] = yy - c.gcb * cb - c.gcr * cr;
        }
    }
}

template <typename T, int YR>
__global__ void k_pack_x2(RifeIoFormat f, Coef c, const T * y0, const T * u0, const T * v0, const T * y1, const T * u1, const T * v1,
    int y_pitch, int c_pitch, const float * ts, __half * out) {
    const int cx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cy = blockIdx.y * blockDim.y + threadIdx.y;
    if (2 * cx >= f.pad_w || YR * cy >= f.pad_h) return;

    float a[YR][2][3], b[YR][2][3];
    if (2 * cx < f.width && YR * cy < f.height) {
        cell_rgb<T, YR>(f, c, y0, u0, v0, y_pitch, c_pitch, cx, cy, a);
        cell_rgb<T, YR>(f, c, y1, u1, v1, y_pitch, c_pitch, cx, cy, b);
    } else {
#pragma unroll
        for (int dy = 0; dy < YR; ++dy)
#pragma unroll
            for (int dx = 0; dx < 2; ++dx)
#pragma unroll
                for (int k = 0; k < 3; ++k) a[dy][dx][k] = b[dy][dx][k] = 0.f;
    }

    const size_t plane = static_cast<size_t>(f.pad_h) * f.pad_w;
    const __half2 t2 = __float2half2_rn(*ts);
#pragma unroll
    for (int dy = 0; dy < YR; ++dy) {
        const size_t o = static_cast<size_t>(YR * cy + dy) * f.pad_w + 2 * cx;   // even: 4-byte aligned half2 stores
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            *reinterpret_cast<__half2 *>(out + k * plane + o) = __floats2half2_rn(a[dy][0][k], a[dy][1][k]);
            *reinterpret_cast<__half2 *>(out + (3 + k) * plane + o) = __floats2half2_rn(b[dy][0][k], b[dy][1][k]);
        }
        *reinterpret_cast<__half2 *>(out + 6 * plane + o) = t2;
    }
}

} // namespace

cudaError_t rife_pack(const RifeIoFormat & f, const RifeFrameIn & f0, const RifeFrameIn & f1,
    const float * d_ts, __half * out, cudaStream_t stream) {
    const dim3 block(32, 8);
    // fast path: horizontal ratio 2, vertical 1 or 2, even padded size (half2 stores)
    if (f.xr == 2 && (f.yr == 2 || f.yr == 1) && (f.pad_w & 1) == 0 && (f.yr == 1 || (f.pad_h & 1) == 0)) {
        const Coef c = make_coef(f);
        const dim3 g((f.pad_w / 2 + block.x - 1) / block.x, (f.pad_h / f.yr + block.y - 1) / block.y);
        if (f.bits > 8) {
            if (f.yr == 2) k_pack_x2<uint16_t, 2><<<g, block, 0, stream>>>(f, c, (const uint16_t *)f0.y, (const uint16_t *)f0.u, (const uint16_t *)f0.v,
                (const uint16_t *)f1.y, (const uint16_t *)f1.u, (const uint16_t *)f1.v, f0.y_pitch, f0.c_pitch, d_ts, out);
            else k_pack_x2<uint16_t, 1><<<g, block, 0, stream>>>(f, c, (const uint16_t *)f0.y, (const uint16_t *)f0.u, (const uint16_t *)f0.v,
                (const uint16_t *)f1.y, (const uint16_t *)f1.u, (const uint16_t *)f1.v, f0.y_pitch, f0.c_pitch, d_ts, out);
        } else {
            if (f.yr == 2) k_pack_x2<uint8_t, 2><<<g, block, 0, stream>>>(f, c, (const uint8_t *)f0.y, (const uint8_t *)f0.u, (const uint8_t *)f0.v,
                (const uint8_t *)f1.y, (const uint8_t *)f1.u, (const uint8_t *)f1.v, f0.y_pitch, f0.c_pitch, d_ts, out);
            else k_pack_x2<uint8_t, 1><<<g, block, 0, stream>>>(f, c, (const uint8_t *)f0.y, (const uint8_t *)f0.u, (const uint8_t *)f0.v,
                (const uint8_t *)f1.y, (const uint8_t *)f1.u, (const uint8_t *)f1.v, f0.y_pitch, f0.c_pitch, d_ts, out);
        }
        return cudaGetLastError();
    }
    const dim3 grid((f.pad_w + block.x - 1) / block.x, (f.pad_h + block.y - 1) / block.y);
    if (f.bits > 8) {
        k_pack<uint16_t><<<grid, block, 0, stream>>>(f, (const uint16_t *)f0.y, (const uint16_t *)f0.u, (const uint16_t *)f0.v,
            (const uint16_t *)f1.y, (const uint16_t *)f1.u, (const uint16_t *)f1.v, f0.y_pitch, f0.c_pitch, d_ts, out);
    } else {
        k_pack<uint8_t><<<grid, block, 0, stream>>>(f, (const uint8_t *)f0.y, (const uint8_t *)f0.u, (const uint8_t *)f0.v,
            (const uint8_t *)f1.y, (const uint8_t *)f1.u, (const uint8_t *)f1.v, f0.y_pitch, f0.c_pitch, d_ts, out);
    }
    return cudaGetLastError();
}

cudaError_t rife_unpack(const RifeIoFormat & f, const __half * in, const RifeFrameOut & o, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 gy((f.width + block.x - 1) / block.x, (f.height + block.y - 1) / block.y);
    const int cw = rife_chroma_w(f), ch = rife_chroma_h(f);
    const dim3 gc((cw + block.x - 1) / block.x, (ch + block.y - 1) / block.y);
    if (f.bits > 8) {
        k_unpack_y<uint16_t><<<gy, block, 0, stream>>>(f, in, (uint16_t *)o.y, o.y_pitch);
        k_unpack_c<uint16_t><<<gc, block, 0, stream>>>(f, in, (uint16_t *)o.u, (uint16_t *)o.v, o.c_pitch);
    } else {
        k_unpack_y<uint8_t><<<gy, block, 0, stream>>>(f, in, (uint8_t *)o.y, o.y_pitch);
        k_unpack_c<uint8_t><<<gc, block, 0, stream>>>(f, in, (uint8_t *)o.u, (uint8_t *)o.v, o.c_pitch);
    }
    return cudaGetLastError();
}

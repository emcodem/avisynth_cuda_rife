// Verifies rife_pack / rife_unpack against zimg reference dumps (see probe scripts in the session notes):
//   argv: src_yuv420p.raw  z_rgb_gbrpf32.raw  z_yv12.raw   (1920x1080)
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "../src/rife_io.h"

static std::vector<uint8_t> readf(const char * p) {
    FILE * f = fopen(p, "rb"); if (!f) { printf("cannot open %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n); fread(v.data(), 1, n, f); fclose(f); return v;
}
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(2); } } while (0)

int main(int argc, char ** argv) {
    const int W = 1920, H = 1080, PW = 1920, PH = 1088;
    auto src = readf(argv[1]); auto zrgb = readf(argv[2]); auto zyuv = readf(argv[3]);
    RifeIoFormat f{ W, H, PW, PH, 8, 2, 2, 0.2126f, 0.0722f, 0 };
    const size_t ysz = (size_t)W * H, csz = (size_t)(W / 2) * (H / 2);

    uint8_t * d_src; CK(cudaMalloc(&d_src, src.size())); CK(cudaMemcpy(d_src, src.data(), src.size(), cudaMemcpyHostToDevice));
    float ts = 0.5f; float * d_ts; CK(cudaMalloc(&d_ts, 4)); CK(cudaMemcpy(d_ts, &ts, 4, cudaMemcpyHostToDevice));
    __half * d_t; CK(cudaMalloc(&d_t, (size_t)7 * PW * PH * 2));
    CK(rife_pack(f, rife_tight_in(f, d_src), rife_tight_in(f, d_src), d_ts, d_t, 0));
    CK(cudaDeviceSynchronize());
    std::vector<__half> h_t((size_t)7 * PW * PH); CK(cudaMemcpy(h_t.data(), d_t, h_t.size() * 2, cudaMemcpyDeviceToHost));

    // zimg gbrpf32 plane order: G, B, R ; ours: R, G, B
    const float * zf = (const float *)zrgb.data();
    const int zi[3] = { 2, 0, 1 };
    double maxd = 0, sum = 0; size_t cnt = 0, over = 0; double padmax = 0;
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < PH; ++y)
            for (int x = 0; x < PW; ++x) {
                float v = __half2float(h_t[(size_t)c * PW * PH + (size_t)y * PW + x]);
                if (y >= H) { padmax = fmax(padmax, fabs(v)); continue; }
                float z = zf[(size_t)zi[c] * W * H + (size_t)y * W + x];
                double d = fabs(v - z); maxd = fmax(maxd, d); sum += d; ++cnt; if (d > 0.002) ++over;
            }
    float tsv = __half2float(h_t[(size_t)6 * PW * PH + 12345]);
    printf("PACK   vs zimg RGBPS: maxdiff=%.6f meanabs=%.8f  px>0.002: %zu  pad_max=%.3g  ts_plane=%.3f\n", maxd, sum / cnt, over, padmax, tsv);
    {   // where is the worst pixel?
        double md = 0; int bx = 0, by = 0, bc = 0;
        for (int c = 0; c < 3; ++c) for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            float v = __half2float(h_t[(size_t)c * PW * PH + (size_t)y * PW + x]);
            float z = zf[(size_t)zi[c] * W * H + (size_t)y * W + x]; double d = fabs(v - z);
            if (d > md) { md = d; bx = x; by = y; bc = c; }
        }
        printf("       worst at c=%d x=%d y=%d diff=%.6f\n", bc, bx, by, md);
    }

    // unpack: feed zimg's own RGB (as fp16) in, compare with zimg's YV12
    std::vector<__half> h_in((size_t)3 * PW * PH, __float2half(0.f));
    for (int c = 0; c < 3; ++c) for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x)
        h_in[(size_t)c * PW * PH + (size_t)y * PW + x] = __float2half(zf[(size_t)zi[c] * W * H + (size_t)y * W + x]);
    __half * d_in; CK(cudaMalloc(&d_in, h_in.size() * 2)); CK(cudaMemcpy(d_in, h_in.data(), h_in.size() * 2, cudaMemcpyHostToDevice));
    uint8_t * d_out; CK(cudaMalloc(&d_out, src.size()));
    CK(rife_unpack(f, d_in, rife_tight_out(f, d_out), 0)); CK(cudaDeviceSynchronize());
    std::vector<uint8_t> o(src.size()); CK(cudaMemcpy(o.data(), d_out, o.size(), cudaMemcpyDeviceToHost));
    const char * nm[3] = { "Y", "U", "V" }; size_t off[3] = { 0, ysz, ysz + csz }, len[3] = { ysz, csz, csz };
    for (int p = 0; p < 3; ++p) {
        int mx = 0; size_t ex = 0;
        for (size_t i = 0; i < len[p]; ++i) { int d = abs((int)o[off[p] + i] - (int)zyuv[off[p] + i]); mx = d > mx ? d : mx; ex += d == 0; }
        printf("UNPACK %s vs zimg YV12: max=%d exact=%.5f%%\n", nm[p], mx, 100.0 * ex / len[p]);
    }
    {   // kernel timing (steady state, 200 iterations each)
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int w = 0; w < 20; ++w) rife_pack(f, rife_tight_in(f, d_src), rife_tight_in(f, d_src), d_ts, d_t, 0);
        cudaEventRecord(e0);
        for (int it = 0; it < 200; ++it) rife_pack(f, rife_tight_in(f, d_src), rife_tight_in(f, d_src), d_ts, d_t, 0);
        cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1);
        printf("TIMING pack   : %.3f ms per call\n", ms / 200);
        for (int w = 0; w < 20; ++w) rife_unpack(f, d_in, rife_tight_out(f, d_out), 0);
        cudaEventRecord(e0);
        for (int it = 0; it < 200; ++it) rife_unpack(f, d_in, rife_tight_out(f, d_out), 0);
        cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1);
        printf("TIMING unpack : %.3f ms per call\n", ms / 200);
    }
    return 0;
}

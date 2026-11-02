# Optimisation potential (measured 2026-11-02)

Scope: RIFE only, optimal quality, no lossy workarounds (no other models, no lower fps, no frame snapping).
Config measured: `deploy/mlrt_trt/example_50p.avs` (rife_v4.17_lite, implementation=2, use_cuda_graph=true,
num_streams=2, Prefetch(8,16), AssumeFPS(30,1)), RTX 5050 8GB, Xeon Gold 6128 (6c/12t), source `D:\kuerbiskern_orig.MP4`.

## Findings

1. **Pipeline speed:** 0.83x realtime on a 600-frame run (0.839x on 1200), consistent with the earlier ~0.87x.
   About 33 RIFE inferences/s are needed at that speed (20% of output frames are exact source frames).
2. **CPU stages are not the limit.** Decode + YUV->RGBPS + pad + crop + YUV conversion alone runs at 175 source fps
   (5.8x realtime); the pipeline needs ~26. Pipeline CPU use during the run: ~21% of 12 threads.
   (Measured separately from RIFE, not concurrently.)
3. **The engine alone is faster than the pipeline.** `trtexec` on `rife_v4.17_lite_fp16.engine`, no data transfers:
   17.2 ms/inference = 58 inferences/s. 2 streams add nothing at that point.
4. **Copies and compute do not overlap on this machine.** Windows per-engine GPU counters in steady state:
   compute engine ~58% busy, copy engine ~41% busy, sum = 99% (= the `nvidia-smi` 99%, which counts any engine
   activity, not SM occupancy). The compute engine is idle ~42% of the time.
   Reproduced outside the plugin: `trtexec` (designed to overlap DMA and compute) gives 33.7 inferences/s with
   1 stream, 2 streams, with/without CUDA graphs, with/without spin-wait. That equals 1 / (17.2 + 8.5 + 3.5 ms).
5. **PCIe link is Gen3 x8 (~7 GB/s)**, although the GPU supports more (host is Skylake-SP Xeon). Not confirmed as
   the cause of the missing overlap.
6. **Per inference the plugin moves (fp32 I/O):** 58 MB in (7 planes: 2 frames x 3 RGB + a constant timestep
   plane = 8 MB wasted) = 8.5 ms; 25 MB out (3 planes) = 3.5 ms. The plugin already uses pinned host memory,
   async copies, CUDA graphs and non-blocking streams, so the transfers are not a plugin bug.

Correction to earlier claims: "GPU is 99% SM-saturated, nothing left to gain" was wrong. The 99% was copy + compute
time, back to back. Overlap is unavailable here, so the lever is moving fewer bytes per inference.

## Optimisation options (estimates from the bandwidth numbers, NOT measured)

Compute stays at 17.2 ms; only transfer time changes.

| Option | Data/inference (in+out) | Transfer | Total | Inferences/s | vs now |
|---|---|---|---|---|---|
| Now (fp32 RGB) | 58 + 25 MB | 12.0 ms | 29.2 ms | 33.7 | - |
| A: 16-bit RGB planes (RGBP16), timestep plane built on GPU | ~25 + 12.5 MB | ~5.5 ms | ~22.7 ms | ~44 | +31% (~1.1x realtime) |
| B: send source YUV 4:2:0 8-bit, convert to RGB on GPU, return 8-bit YUV | ~6.2 + 3.1 MB | ~1.4 ms | ~18.6 ms | ~54 | +60% (~1.3x realtime) |
| B': same as B for 10-bit 4:2:2 (16-bit containers) | ~16.7 + 8.4 MB | ~3.6 ms | ~20.8 ms | ~48 | +42% (~1.2x realtime) |

Hard ceiling with zero transfer cost: 58 inferences/s (~1.45x realtime at 50 fps).

- **A** needs: plugin accepts 16-bit integer planar RGB input/output, CUDA kernel converts to/from the engine's
  fp32 tensors and fills the timestep plane. Script passes `RGBP16`. 16-bit is effectively lossless for 8/10-bit sources.
- **B** needs: CUDA kernels for YUV->RGB (matrix, range, chroma upsampling) and the reverse. Must be validated
  against zimg (`z_ConvertFormat`) output on real footage (PSNR/SSIM plus visual check) before use; chroma
  resampling kernel and rounding/dithering must match or beat zimg.

## Update: fp16 I/O measured, and the chosen direction

- Measured with `trtexec` on an engine built from the same ONNX with `--inputIOFormats=fp16:chw --outputIOFormats=fp16:chw`:
  42.3 inferences/s with transfers (H2D 4.3 ms, D2H 1.8 ms, compute 17.3 ms) vs 33.7 now = +26%. Estimate confirmed.
- Chosen direction: raw planar YUV in/out with GPU-side conversion, aligned with the token-frame design of
  emcodem/mvtools `gpu_work` (`cudafilters`). Required mvtools_cuda additions are in `MVTOOLS_CUDA_CHANGES.md` (treated as
  already made). The plugin also keeps a standalone path (own pinned YUV upload + own kernels, no mvtools needed) and gets
  a `cudadownload` parameter (default true).

## 10-bit 4:2:2 support

Yes, both options can support it.
- **A:** already works at the script level, since zimg converts any source (incl. YUV422P10) to RGBP16.
  The plugin only has to take 16-bit planes.
- **B:** the GPU kernel reads planar 16-bit containers (YUV422P10 / P210) and upsamples chroma horizontally only
  (4:2:2 is simpler than 4:2:0). Output goes back to 10-bit 4:2:2. Needs per-source matrix/range selection
  (e.g. BT.709 vs BT.2020) and a 10-bit-correct rounding/dither step. fp16 engine output has ~11 bits of mantissa,
  enough for 10-bit but worth checking in gradients/banding tests.
- 10-bit 4:2:2 moves ~2.7x the bytes of 8-bit 4:2:0, so B' gains less than B (see table).

## Other (smaller) levers, not yet tried
- `implementation=2` on a full (non-lite) model for the quality baseline (lite measured ~27 dB PSNR vs full v4.17).
- `trtexec --dumpProfile` per-layer profile; `builderOptimizationLevel=5` + timing cache; ONNX graph cleanup.

## Not worth pursuing (given the above)
- More streams / batching: compute is 17.2 ms per frame at full-frame size; 2 streams give no gain in `trtexec`,
  and `num_streams>=3` was far worse in the pipeline.
- Rewriting the inference engine: TensorRT already fuses and tunes; the loss here is transfer time, not kernels.

## Reproduce
Scripts and logs in the session scratchpad (`full.avs`, `cpuonly.avs`, `engines.ps1`, `gpu_full.csv`, `eng_samples.txt`).
Commands: `ffmpeg -i full.avs -frames:v 600 -f null -` (add `MLRT_TRT_TIMING=1` for per-call timing),
`trtexec --loadEngine=<engine> --iterations=300 --warmUp=500 --duration=6 [--noDataTransfers] [--streams=N] [--useCudaGraph]`,
PowerShell `Get-Counter '\GPU Engine(*)\Utilization Percentage'` filtered by the ffmpeg PID.

---

# Results of the optimisation work (2026-11-02)

## Where it ended up (RTX 5050, 1080p29.97 -> 50p, rife_v4.17_lite / implementation=2, real footage)

| Stage | Realtime speed |
|---|---|
| Starting point: zimg YUV->RGBPS, fp32 I/O engine, CPU AddBorders/Crop | 0.84x |
| + GPU-side YUV<->RGB-fp16, padding and crop in CUDA (raw YUV over PCIe), fp16-I/O engine | 1.23x |
| + one shared filter instance (MT_NICE_FILTER) instead of one per Prefetch thread | 1.25x |
| + fast 2x2-block pack kernel (0.77 ms -> 0.15 ms) | **1.28-1.29x** (+54%) |

Compute-only ceiling (engine alone, no I/O at all) is 58 inferences/s = ~1.45x.

## Where the time goes now (GPU timeline, `MLRT_TRT_TIMELINE=1`; ms per inference)
H2D raw YUV 0.91 | pack 0.15 | TensorRT 17.5 | unpack 0.39 | D2H 0.52 -> ~19.5 ms.
- **The GPU runs exactly one call at a time (100% of the time).** The host does have two calls in flight (95% of the time
  two threads are between ticket-acquire and GPU-sync) but the driver executes the two streams strictly one after the other:
  copies never overlap compute, there are no idle gaps. So num_streams 1 vs 2 vs 4 changes nothing once the GPU is fed
  (num_streams=1 is slower, 1.0x, only because the CPU copy phases then sit on the critical path).
- 90% of the chain is TensorRT. Nothing at the plugin level can shrink that.

## Corrections to earlier claims
- "GPU is 99% SM-saturated" -- it was copy + compute back to back (see top of this file).
- "num_streams=3 / Prefetch(12,24) cause GPU contention / thread oversubscription" -- wrong. The plugin was registered
  MT_MULTI_INSTANCE, so every Prefetch thread built its own filter instance with its own engine copy and num_streams execution
  contexts: ~5.3 GB of VRAM at Prefetch(8,16) on an 8 GB card; one step more spilled into WDDM paging (10-100x slower).
  With MT_NICE_FILTER (one shared instance) any num_streams / Prefetch value is fine.

## Tried, did not help
- Engine build variants (optimization level 5, all tactic sources): 57.4 / 58.3 / 57.9 / 57.9 inferences/s -> noise. TensorRT is at
  the floor for this network.
- Zero-copy (kernels reading/writing pinned host memory over UVA instead of memcpy): output only = no change, input = much
  slower (0.87x), both = 0.82x. Kept as an experiment: MLRT_TRT_ZEROCOPY=o|i|io (default off).
- CUDA graphs: ~1% (kept on). Prefetch depth: flat from (4,8) to (16,32).

## Quality / exactness
- GPU YUV<->RGB matches zimg: Y bit-exact (round to nearest), chroma 99.9% exact (max 1 LSB), RGB within fp16 precision.
  Findings that were needed to get there: Catmull-Rom (b=0, c=0.5) both ways, chroma left co-sited / vertically centred, and
  **half-sample mirror edges** (clamping was wrong: up to 4.5e-3 error at the borders).
- Pass-through frames (20% of the 3:5 output) come out bit-identical to the source (old RGB round-trip: 96.3% exact).
- Interpolated frames vs the old pipeline: mean 49.0 dB PSNR; the old pipeline against itself with a re-built engine: 52.2 dB.
  So the difference sits at the engine-to-engine numerical noise of RIFE (differences concentrate at borders, max 130 levels
  at single pixels). Bit-exactness with the old path is not reachable (different engine build, fp16 I/O) and is not needed.
- Also verified: YUV422P10, YUV444, 1916x1076 (non-multiple-of-32, pad+crop), sc=true; legacy RGB path unchanged (bit-identical).

## Remaining headroom (small)
- Device-side frame cache so each source picture is uploaded once instead of ~2.5x (H2D 0.91 -> ~0.35 ms: +2.7%);
  mvtools_cuda's keyed upload does this automatically on the token path.
- Specialised unpack chroma kernel (0.39 -> ~0.25 ms: +0.8%).
- Windows hardware-accelerated GPU scheduling (HAGS) might let copies overlap compute; HwSchMode is not set in the registry
  here so its state is unknown (a machine setting, not a code change; needs a reboot and a re-run of MLRT_TRT_TIMELINE).
- FP8/INT8 engine: numerics change, not tried.

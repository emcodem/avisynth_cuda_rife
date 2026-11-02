# mlrt_trt -- portable AviSynth+ TensorRT RIFE plugin

Self-contained deployment package. Drop the contents of this folder's top
level into any single folder -- your AviSynth+ `plugins` directory, or
wherever ffastrans loads AviSynth plugins from. Nothing else needs to be
installed: no Visual C++ redistributable, no CUDA Toolkit, no TensorRT SDK.
The only external requirement is an NVIDIA GPU with a driver already
installed (the driver ships `nvcuda.dll`, a Windows system component in the
OS/driver layer, not something an app bundles).

Source material can be **any resolution** -- see "Two ways to get an engine"
below. A static TensorRT engine only runs at the exact frame size it was
built for, so arbitrary resolutions are handled by building (and caching) an
engine on first use, the same way `vsmlrt.py` itself does it upstream.

## What's in here and why it's enough

| File | Needed for | Why nothing else is |
|---|---|---|
| `mlrt_trt.dll` | the plugin itself | statically linked against the VC++ CRT (`/MT`) -- its only direct import is `KERNEL32.dll` (verified via `dumpbin /dependents`) |
| `nvinfer_10.dll` | TensorRT 10.16 core runtime | inference (loading + running) |
| `cudart64_13.dll` | CUDA 13.x runtime | inference |
| `trtexec.exe` | building an engine from ONNX | on-demand builds only |
| `nvonnxparser_10.dll`, `nvinfer_plugin_10.dll` | `trtexec`'s own dependencies | on-demand builds only |
| `nvinfer_builder_resource_sm120_10.dll` | kernel generation for **this GPU's architecture** (Blackwell / `sm_120`) | on-demand builds only, and **only for this GPU arch** -- see caveat below |
| `models/rife/*.onnx` | on-demand engine builds | -- |
| `models/rife/*_fp16.engine` | pre-built engines for the one resolution this project tested at (1920x1088, i.e. 1080p padded) | skips the ~30-90s first-build cost for that one resolution/this exact GPU -- see caveat below |

Deliberately **not** included, confirmed unnecessary by testing: cuBLAS,
cuBLAS_LT, cuDNN, `nvrtc*.dll`, TensorRT-RTX, `cupti`, MIGraphX. If you only
ever need inference (never a fresh on-demand build -- i.e. every resolution
you'll ever see already has a cached `.engine`), you can strip this down to
just the first three files in the table above; that was the original,
leaner ~570MB variant of this package. With build-on-demand support this
package is ~984MB, dominated by `nvinfer_10.dll` (376MB) and
`nvinfer_builder_resource_sm120_10.dll` (250MB).

**How the portability actually works:** normally a DLL's dependencies are
searched relative to the *host application's* directory (ffmpeg.exe,
AvsPmod.exe, ffastrans's own binary, ...), not the plugin's own folder --
standard Windows DLL search order, not a bug. `mlrt_trt.dll` works around
this explicitly: at filter-creation time it resolves its own module path and
calls `LoadLibraryExA(..., LOAD_WITH_ALTERED_SEARCH_PATH)` on both
`cudart64_13.dll` and `nvinfer_10.dll` using that path, before touching
CUDA/TensorRT anywhere else. `trtexec.exe`, by contrast, is spawned as its
own process (not loaded as a library into a host), so for it the classic
search order already resolves everything correctly as long as its
dependencies sit in the *same folder as `trtexec.exe` itself*. Net effect:
**every file in this folder must stay in this one folder together** (no
subfolders for the DLLs/exe) -- but the folder itself can be anywhere, and
whatever loads `mlrt_trt.dll` can be anything. Verified by running the exact
same `.avs` script through two unrelated `ffmpeg` builds in two different
directories with no shared files.

## CRITICAL: both the GPU architecture DLL and pre-built engines are tied to this machine

This machine: RTX 5050 (Blackwell, `sm_120`), driver 591.86, CUDA 13.1,
TensorRT 10.16.0. Two consequences:

1. **The bundled `*_fp16.engine` files** are compiled plans tied to that
   exact combination -- they will fail to load (or silently pick wrong
   tactics) on a different GPU architecture, driver, or TensorRT version. If
   a render node differs, either delete them (on-demand building will
   recreate an equivalent one, cached, on first use) or rebuild them
   explicitly ahead of time with `trtexec.exe` (see its invocation in
   `mlrt_trt.cpp`'s `resolveOrBuildEngine()`, or just let the plugin do it
   once and copy the resulting cache file out).
2. **`nvinfer_builder_resource_sm120_10.dll`** only contains kernel-generation
   resources for `sm_120`. If render nodes use a *different* GPU
   architecture, on-demand builds will fail there until you swap in (or add)
   the matching `nvinfer_builder_resource_sm<XX>_10.dll` for that
   architecture, from the same TensorRT release. Inference of an
   already-built engine doesn't need this file at all -- only building does.

If every ffastrans node is this same machine/image, none of this matters and
everything here works as-is.

## Two ways to get an engine

**A. Point at a pre-built `.engine` file (fastest, needs a matching GPU):**
```avs
c = mlrt_RIFE_trt(c, "<this folder>\models\rife\rife_v4.17_fp16.engine", 1, factor_num=2, factor_den=1, device=0, num_streams=2)
```
Only works at exactly the resolution/GPU/TensorRT version it was built for
(1920x1088 padded, this machine, for the bundled engines).

**B. Point at the `.onnx` model instead, for arbitrary resolutions:**
```avs
c = mlrt_RIFE_trt(c, "", 1, onnx_path="<this folder>\models\rife\rife_v4.17.onnx", factor_num=2, factor_den=1, device=0, num_streams=2)
```
Pass `""` for `engine_path` (or a path that doesn't exist yet). The first
time a given clip resolution + GPU + TensorRT version combination is seen,
`mlrt_trt` shells out to `trtexec.exe` (found next to `mlrt_trt.dll`, or via
an explicit `trtexec_path=` argument) to build a matching engine, saves it as
`<onnx-dir>\rife_v4.17_<gpu-name>_trt<version>_<W>x<H>_fp16.engine`, and
reuses that cached file on every later call at the same resolution/hardware
-- no rebuild. Cold build cost: ~30-90s (one-time per resolution/GPU/TRT
version); warm/cached calls are indistinguishable from mode A. Two ffastrans
jobs racing to build the exact same new-resolution engine at once won't
corrupt each other -- each builds to a private temp file first and only the
first to finish gets renamed into place, the other discards its redundant
copy and reuses the winner's file.

This mirrors what `vsmlrt.py` itself does upstream (`get_engine_path()` /
`trtexec()`); ffastrans nodes seeing arbitrary source material should
default to mode B and let the cache warm up naturally as new resolutions are
encountered.

## Usage

**See `example_50p.avs` in this folder for a complete, runnable, self-contained
template of the current recommended config** (`rife_v4.17_lite`/`implementation=2`,
`AssumeFPS`-based retiming speedup, `use_cuda_graph=true`, the duration fix, tuned
`Prefetch`/`num_streams`) -- copy it and swap in your own source clip. It resolves
every path inside this package via `ScriptDir()`, so the folder can be copied
anywhere and the script still finds its own plugin/models. Encode it with:
```
ffmpeg -i example_50p.avs -vf fps=50 -c:v <your codec> output.mp4
```
The `-vf fps=50` is required to get a literal 50.000fps CFR output with the
correct total duration -- see "Duration fix" below for why.

Minimal/manual usage, for reference or a different model/resolution:
```avs
LoadPlugin("<this folder>\mlrt_trt.dll")
Import("<this folder>\scripts\mlrt_RIFE_trt.avsi")

c = <your source clip, converted to 32-bit planar RGB, width/height padded to a multiple of 32>
c = mlrt_RIFE_trt(c, "", 1, onnx_path="<this folder>\models\rife\rife_v4.17.onnx", factor_num=2, factor_den=1, device=0, num_streams=2)
c = <crop padding back off, convert back to your target format>
Prefetch(c, 4, 8)
```

See `scripts/mlrt_RIFE_trt.avsi`'s header comment for the full parameter
list (including `overlap_w/h`, `tilesize_w/h`, `verbosity`, `sc`/`sc_threshold`,
and `trtexec_path` for overriding where `trtexec.exe` is found).
`implementation=1` (the second positional arg, `1` above) requires mod-32
width/height for all bundled RIFE versions (v4.9/4.17/4.18/4.20); RIFE
itself doesn't support tiling (upstream `vsmlrt.py`'s `RIFEMerge` explicitly
rejects it -- optical-flow estimation needs full-frame context), so pad to
the nearest multiple of 32 rather than trying `overlap_w`/`tilesize_w`.
`num_streams=2` and `Prefetch(4,8)` are the tuned values for a
6-core/12-thread host + 8GB-class GPU (see the main project handoff doc for
how these were derived -- re-tune if the target hardware differs
significantly; `example_50p.avs` uses the further-retuned `Prefetch(8,16)`
that config needs, see below).

## Model versions included

`rife_v4.9`, `rife_v4.17`, `rife_v4.17_lite`, `rife_v4.18`, `rife_v4.20` --
all implementation-1 (11-plane), fp16, verified visually clean on real
footage (no interpolation artifacts) and functionally verified end-to-end
through this exact plugin, both from a pre-built engine and via on-demand
building. `rife_v4.4` is deliberately excluded: it produces a checkerboard
artifact on real (non-flat) footage on this hardware/TensorRT combination --
see the main project handoff doc for details.

**`rife_v4.17_lite` is the fastest verified-clean model in this package**
(~0.58x realtime on this GPU/resolution vs. ~0.55x for `v4.9` and ~0.48x for
full `v4.17`, measured end-to-end on real footage -- see the handoff doc's
"Fastest known-clean RIFE model" section for the full benchmark table and
correctness-check methodology). Only checked on a low-motion clip segment so
far, not yet stress-tested on high-motion footage. Use it the same way as
any other bundled model:
```avs
c = mlrt_RIFE_trt(c, "<this folder>\models\rife\rife_v4.17_lite_fp16.engine", 1, factor_num=2, factor_den=1, device=0, num_streams=2)
```
Prefer full `rife_v4.17` instead if you need the extra fidelity and can
afford the ~18% slower throughput.

**Faster still: `implementation=2` (7-plane, internal-padding RIFE graph),
previously untested end-to-end in this project.** `models\rife_v2\` ships a
`rife_v4.17_lite` engine built from the 7-plane graph instead of the 11-plane
one `models\rife\` uses. It skips the positional-encoding/multiplier input
planes entirely, which cuts raw GPU compute time by ~35% (`trtexec`'s own
numbers: 17.3ms vs 26.7ms) and measured **0.683x-0.688x realtime**
end-to-end on this GPU/resolution -- the fastest verified-clean config found
in this project, ~17% faster than the same model via `implementation=1` and
~42% faster than full `rife_v4.17` via `implementation=1`. Verified visually
clean the same way as the other models (full-frame + contrast/sharpen zoom
crop, no checkerboard/grid artifact); PSNR ~28.6dB/SSIM~0.95 vs. the
`implementation=1` output on the same frames -- a real but non-artifact-shaped
difference (different graph, not a broken one). Same "only checked on a
low-motion segment" caveat as `v4.17_lite` above applies here too. Usage:
```avs
c = mlrt_RIFE_trt(c, "<this folder>\models\rife_v2\rife_v4.17_lite_fp16.engine", 2, factor_num=2, factor_den=1, device=0, num_streams=2)
```
Note `implementation=2` doesn't need mod32 padding (the network pads
internally) -- the `AddBorders`/`Crop` steps in the usage example above are
harmless but no longer required if you switch to this graph; native
resolution wasn't tested here, so pad-and-crop is still the safer default
until that's confirmed. (Update: native 1920x1080 *was* tested in a later
session -- came back marginally slower than 1088-padded, not a win, see the
main handoff doc.)

## Current fastest verified-clean, no-quality-cost config (as of 2026-08-13)

Confirmed clean (each measured in its own isolated process, no concurrent
GPU/CPU contention -- see the main handoff doc for why that matters):
relabel the source framerate (29.97 -> 30, via `AssumeFPS`) before the RIFE
retiming math -- raises the rate of output frames that land exactly on a
source frame (and can skip RIFE entirely) from ~0.1% to 20%, at the cost of
a ~0.1% timing/duration compression this needs correcting for (see "Duration
fix" below) -- plus `use_cuda_graph=true` and a `Prefetch` retune for the
resulting mixed-cost (light-skip / heavy-RIFE) workload:

```avs
c = FFVideoSource(...)
c = AssumeFPS(c, 30, 1)
c = <convert to 32-bit planar RGB, pad to mod32>
c = mlrt_RIFE_trt(c, "<this folder>\models\rife_v2\rife_v4.17_lite_fp16.engine", 2, fps_num=50, fps_den=1, device=0, num_streams=2, use_cuda_graph=true)
c = AssumeFPS(c, 50000, 1001)   # duration fix -- see below, required before encode
c = <crop padding back off, convert back to target format>
Prefetch(c, 8, 16)
```

Measured **0.868-0.871x realtime** at a 50fps target on this GPU/resolution,
up from 0.683-0.688x (`implementation=2` alone) and 0.481x (session-start
baseline, `implementation=1`, no `AssumeFPS`/`use_cuda_graph`). `num_streams`
must stay at **2** -- `num_streams=3` was retested cleanly and collapsed to
~0.13x realtime, confirming genuine GPU-side contention on this 8GB card
above 2 concurrent streams, not a benchmarking artifact.

**Duration fix, required if the deliverable needs both literal 50.000fps CFR
and correct duration (2026-08-13):** `AssumeFPS(30,1)` compresses the
retiming clock by 1000/1001 vs. the source's true 29.97fps, so a naively
50/1-labeled output ends up ~0.1% (~34ms on the 34s test clip) short of the
true source duration. Relabeling the AviSynth output at `50000/1001` (shown
in the snippet above) cancels that losslessly, but the result is then
nominally ~49.950fps, not 50 -- fine if only duration matters, not fine if
the container must say exactly 50.000. **Get both** by doing the final rate
conversion downstream at encode time with ffmpeg's `fps` filter:
```
ffmpeg -i script.avs -vf fps=50 -c:v <codec> ...
```
This inserts/drops frames at the mathematically correct sparse positions
(here, exactly one duplicated frame per 1000 output frames, ~once every 20s)
to hit true 50.000fps CFR while preserving the corrected timeline -- the same
technique broadcast pipelines use for 23.976<->24 / 29.97<->30 conversions.
Validated on the full 1021-frame test clip: `ffprobe` on the result showed
`r_frame_rate=50/1`, 1703 frames, 34.060s total duration vs. the source's
true 34.067367s -- **residual gap ~7.4ms** (down from ~34ms uncorrected),
the unavoidable quantization of correcting in whole 1/50s frame steps on a
fixed-CFR grid, well under any perceptible A/V sync threshold. Skip the
`AssumeFPS(c, 50000, 1001)` relabel (and just use a plain 50/1 label) if the
deliverable doesn't care about matching the source's exact original
duration -- e.g. a picture-only pass with no audio to keep in sync.

Several other approaches (fp16 engine I/O, batched inference, TensorRT-RTX,
INT8, an exhaustive `trtexec` builder-flag sweep) were tried and ruled out --
see `PORTING_TO_AVISYNTH.md`'s "Dead ends" list.

**Rejected: escalated snap-tolerance variant.** A further wrapper that snaps
interpolation positions within one "tick" of a source frame (not just exact
matches) to that source frame measured **1.71-1.72x realtime** -- but
introduces up to ~6.7ms of per-frame timing error on the snapped frames.
Deliberately not adopted: that's too large a quality risk for an unverified
gain, so it's excluded from this package. Prototype kept at
`test/mlrt_RIFE_trt_snap.avsi` for reference only if this gets revisited with
a tighter tolerance and real visual verification on high-motion footage.

This config is not yet copied into `scripts/mlrt_RIFE_trt.avsi`/this
package's models -- `models/rife_v2/rife_v4.17_lite_fp16.engine` (already
bundled) is the right engine, but the calling script needs the `AssumeFPS`,
`use_cuda_graph=true`, and `Prefetch(8,16)` changes shown above applied by
whoever wires this into the production ffastrans pipeline. See
`PORTING_TO_AVISYNTH.md`'s "Session (2026-08-12/13)" section and
`test/test_real_video.avs` (updated to this config) for the full story.

---

## YUV fast path (added 2026-11-02): GPU-side YUV<->RGB, ~1.28x realtime instead of ~0.84x

Pass a planar integer YUV clip (YV12, YUV422P10, YUV444P16, ...; 8..16 bit) to `mlrt_RIFE_trt` instead of fp32 RGB and the plugin
does YUV->RGB, the padding to the engine tile and the crop back on the GPU. Only raw YUV crosses PCIe (6 MB in / 3 MB out per
inference instead of 58 MB / 25 MB). See `example_50p_yuv.avs`. Requirements: `implementation=2`, an **fp16-I/O engine**
(`models\rife_v2\rife_v4.17_lite_fp16io.engine`, or `onnx_path=` to build one on demand: engines get `_fp16io` in their name),
no tiling. Output has the input's pixel type and size. The old RGB interface is unchanged (bit-identical output).

Extra parameters (passed through `mlrt_RIFE_trt` to `mlrt_trt`): `matrix` ("709" default / "601" / "2020"), `range`
("limited" default / "full"), `pad_multiple` (32), `cudadownload` (true), `mvtools_cuda` (true). Chroma is assumed sited like
MPEG-2/H.264 (left, vertically centred); conversions match zimg (Catmull-Rom, mirrored edges). `sc_threshold` is in 8-bit luma
units in YUV mode.

`cudadownload` / mvtools_cuda token frames: see `..\..\MVTOOLS_CUDA_CHANGES.md`. Without mvtools_cuda.dll nothing changes: the
plugin runs standalone (own pinned raw-YUV upload + own kernels). With a v20 mvtools_cuda next to it, it takes token frames
(`CudaUpload`) and returns token frames (`cudadownload=false`, then `CudaDownload`) or normal frames (`cudadownload=true`).
Implemented against the spec and verified with a mock library only.

Other changes in this build: the filter is now registered `MT_NICE_FILTER` (one shared instance for all Prefetch threads; it was
one instance + engine copy + contexts PER thread, which used ~5 GB of VRAM and fell off a cliff at num_streams=3 or
Prefetch(12,..) on an 8 GB card -- those settings are fine now). Diagnostics: `MLRT_TRT_TIMING=1` (per call host timings),
`MLRT_TRT_TIMELINE=1` (GPU event timeline per phase; use with use_cuda_graph=false), `MLRT_TRT_ZEROCOPY=o|i|io` (experiment,
slower). Build: `avs-mlrt\mlrt_trt\tools\build_plugin.bat` (VS2022 + the nvcc from `_cuda_headers_env`); kernel check against
zimg dumps: `tools\test_rife_io.cu`; mock library: `tools\mock_mvtools_cuda.cpp`.
Previous plugin / wrapper are kept in `D:\dev\avs_rife_tensor\backup_pre_yuv\`.

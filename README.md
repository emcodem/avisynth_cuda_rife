# avisynth_cuda_rife

Fork of [Asd-g/avs-mlrt](https://github.com/Asd-g/avs-mlrt) that adds **`mlrt_trt`: RIFE frame interpolation for AviSynth+ on NVIDIA GPUs through TensorRT**, with a GPU-side YUV path that keeps the pipeline off the PCIe bus and an optional zero-copy hand-over to the GPU filters of [emcodem/mvtools (`gpu_work`, `cudafilters`)](https://github.com/emcodem/mvtools/tree/gpu_work/cudafilters).

The original ncnn / OpenVINO / ONNX Runtime plugins are untouched; everything added lives in [`mlrt_trt/`](mlrt_trt) and [`docs/`](docs). The upstream README is kept below.

## What `mlrt_trt` adds

- **TensorRT backend** for the avs-mlrt plumbing (engine loading, on-demand engine build + cache through `trtexec`, `mlrt_RIFE_trt` wrapper with integer and fractional frame-rate conversion, scene-change option). Its engine handling follows [vs-mlrt](https://github.com/AmusementClub/vs-mlrt).
- **YUV fast path**: feed a planar integer YUV clip (8..16 bit; 4:2:0, 4:2:2, 4:4:4, e.g. `YV12` or `YUV422P10`). YUV->RGB, padding to the engine tile and the crop back to the clip size run in CUDA kernels on the GPU. Only raw YUV crosses PCIe (about 6 MB in / 3 MB out per inference instead of 58 MB / 25 MB of fp32 RGB), and no `z_ConvertFormat` / `AddBorders` / `Crop` is needed in the script.
- **Shared filter instance** (`MT_NICE_FILTER`): one engine copy for all `Prefetch` threads. The earlier one-instance-per-thread registration used about 5 GB of VRAM on an 8 GB card and fell off a cliff at `num_streams=3` or `Prefetch(12,...)`.
- **mvtools_cuda device-frame interop** (token frames in/out, `cudadownload` parameter), see below.
- The old fp32 planar RGB interface still works and gives bit-identical output.

Measured on an RTX 5050 (8 GB), 1080p 29.97 -> 50p, `rife_v4.17_lite`, `implementation=2`, real footage:

| Pipeline | Realtime speed |
|---|---|
| RGB path (zimg conversions, fp32 I/O) | 0.84x |
| YUV fast path | **~1.28x** (compute-only ceiling of the engine: ~1.45x) |

The YUV path matches the zimg conversions it replaces (Y bit-exact, chroma 99.9% exact within 1 level; Catmull-Rom chroma resampling, MPEG-2 chroma siting, mirrored edges). Pass-through frames come out bit-identical to the source. Interpolated frames differ from the RGB path at the level two builds of the same TensorRT engine differ from each other (~49 dB PSNR vs ~52 dB). Details, the timeline analysis and what did not help: [docs/optimisation_potential.md](docs/optimisation_potential.md).

## Usage

```avisynth
LoadPlugin("mlrt_trt.dll")                 # + cudart64_13.dll, nvinfer_10.dll next to it
LoadPlugin("ffms2.dll")
Import("mlrt_RIFE_trt.avsi")               # mlrt_trt/scripts/

c = FFVideoSource("clip.mp4")              # any planar integer YUV, no conversion needed
c = AssumeFPS(c, 30, 1)                    # 29.97 -> 30: 20% of the 50p output frames are exact source frames and skip RIFE
c = mlrt_RIFE_trt(c, "rife_v4.17_lite_fp16io.engine", 2, fps_num=50, fps_den=1, \
        num_streams=2, use_cuda_graph=true)
c = AssumeFPS(c, 50000, 1001)              # duration-correct relabel
Prefetch(c, 8, 16)
```

Encode with ffmpeg's `fps` filter to get true 50.000 fps CFR: `ffmpeg -i script.avs -vf fps=50 ...`. More complete examples: [`mlrt_trt/examples/`](mlrt_trt/examples) (`example_50p_yuv.avs` = fast path, `example_50p.avs` = RGB path). The parameter reference, engine building and deployment notes are in [`mlrt_trt/README.md`](mlrt_trt/README.md).

Key parameters of `mlrt_RIFE_trt(clip, engine_path, implementation, ...)`:

| Parameter | Meaning |
|---|---|
| `engine_path` / `onnx_path` | A pre-built engine, or an ONNX file to build and cache one on first use (30-90 s, needs `trtexec.exe` next to the plugin). |
| `implementation` | `2` (RIFE graph with internal padding, 7 input planes) is required in YUV mode; `1` only on the RGB path. |
| `fps_num`, `fps_den` / `factor_num`, `factor_den` | Target frame rate / factor. Any rational ratio, also decreasing. |
| `num_streams`, `use_cuda_graph` | `2` / `true` is what we use; the GPU executes one inference at a time anyway. |
| `matrix`, `range` | YUV mode: `"709"` (default) / `"601"` / `"2020"`; `"limited"` (default) / `"full"`. |
| `pad_multiple` | YUV mode: engine tile granularity (default 32). |
| `cudadownload` | YUV mode with mvtools_cuda: `true` (default) returns normal frames, `false` returns token frames. |
| `sc`, `sc_threshold` | Scene change detection (threshold in 8-bit luma units in YUV mode). |

The low-level filter is `mlrt_trt([clipa, clipb, timestep], engine_path=..., ...)`; it additionally has `mvtools_cuda` (default `true`) to disable the mvtools_cuda interop. In YUV mode the engine must have **fp16 input/output tensors** (`trtexec --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw`); with `onnx_path` the plugin builds it that way and tags the cached file `_fp16io`.

## Working with mvtools GPU filters (token frames)

[`emcodem/mvtools` branch `gpu_work`](https://github.com/emcodem/mvtools/tree/gpu_work) keeps frames on the GPU between filters ("token frames", `CudaUpload` / `CudaDownload`, `mvtools_cuda.dll`). `mlrt_trt` is written to take part in that design:

```avisynth
c = CudaUpload(c)                                   # cudafilters
c = mlrt_RIFE_trt(c, engine, 2, fps_num=50, fps_den=1, cudadownload=false)   # token frames in AND out
c = CudaDownload(c)
```

- Token frames in are used as they are, normal frames are uploaded through the library's keyed upload (once per picture). With `cudadownload=true` (default) the filter returns normal frames, with `false` it returns token frames, as the other GPU filters do.
- **Needs `mvtools_cuda` API version 20.** The current `gpu_work` library (API 19) cannot hand out the device pointer of a token or create a device frame from outside; the five functions that have to be added (`mvcuda_token_acquire/release`, `mvcuda_frame_alloc/commit/discard`) are specified in [docs/MVTOOLS_CUDA_CHANGES.md](docs/MVTOOLS_CUDA_CHANGES.md).
- **Status:** the plugin side is implemented and verified against a mock library only (CPU in, token in -> token out, token in -> CPU out are bit-identical to the standalone path). Not yet tested against a real v20 library: event ordering, eviction/regeneration under a small cache budget, mixing the CUDA 12 runtime of mvtools_cuda with the CUDA 13 runtime used here.
- If `mvtools_cuda.dll` is missing, has no GPU, or reports another API version, `mlrt_trt` silently runs standalone (own pinned raw-YUV upload and kernels). Nothing else is required.
- All plugins of a process must share one `mvtools_cuda.dll` (the frame registry is per library instance): `mlrt_trt` uses the instance already loaded by `mvtools2` / `cudafilters`, otherwise it loads one from its own folder.

## Compatibility and limits

- **Windows x64 only** for `mlrt_trt` (on-demand engine build and the mvtools_cuda interop use Win32 APIs). **NVIDIA GPU** required. Tested on an RTX 5050 (Blackwell, `sm_120`) with TensorRT 10.16 and the CUDA 13 runtime; the kernels are built for sm_75/86/89/90/120 (+ PTX for 12.0).
- **AviSynth+ r3928 or later** (C plugin API, interface version 10).
- **TensorRT engines are tied to the GPU architecture, driver and TensorRT version** they were built with; keep the ONNX files and let the plugin rebuild (`onnx_path`) on other machines. The shipped runtime DLLs must sit in one folder with `mlrt_trt.dll`.
- **YUV mode limits:** `implementation=2` only, no tiling (`overlap_*` / `tilesize_*`), engine tile >= clip size, planar integer YUV only (no YUY2/packed, no float YUV, no RGB), BT.709 / 601 / 2020 non-constant-luminance matrices, chroma assumed sited like MPEG-2 / H.264 (left, vertically centred). Frames on the GPU are kept at 8/16-bit integer precision; the network runs in fp16.
- **Models:** ONNX RIFE models come from the [vs-mlrt model releases](https://github.com/AmusementClub/vs-mlrt/releases); `implementation=2` needs the internal-padding ("rife_v2") graphs. Check new RIFE versions on real, detailed footage before adopting them: a flat synthetic clip hides broken warp kernels (we saw this with the `rife_v4.4` export of the bundle, which produces a grid pattern on real footage).
- **Throughput:** the GPU executes one inference at a time (no overlap of copies and compute under the Windows driver, no idle gaps), so `num_streams=2` is enough and TensorRT is ~90% of the time per frame. Any speed beyond this has to come from the model or the hardware.
- **Diagnostics:** `MLRT_TRT_TIMING=1` (host timings per call), `MLRT_TRT_TIMELINE=1` (GPU phase timeline; use with `use_cuda_graph=false`, analysis scripts in `mlrt_trt/tools/`).

## Building `mlrt_trt`

Visual Studio 2022, CMake/Ninja, TensorRT 10.x SDK, CUDA runtime headers + `cudart.lib`, AviSynth+ C headers + import lib, and an `nvcc` (13.x) for `src/rife_io.cu`. [`mlrt_trt/tools/build_plugin.bat`](mlrt_trt/tools/build_plugin.bat) shows the layout it expects (`DEPS_ROOT`, `NVCC`, `NVCC_INCLUDE` can be overridden). `mlrt_trt` does not need protobuf / onnx / boost, so it builds standalone with its own `mlrt_trt/CMakeLists.txt` and does not go through the root CMake. `tools/test_rife_io.cu` checks the conversion kernels against zimg reference frames, `tools/mock_mvtools_cuda.cpp` is the mock library used for the token-frame tests.

## Credits

- [Asd-g/avs-mlrt](https://github.com/Asd-g/avs-mlrt): the AviSynth+ port of vs-mlrt this fork is based on (GPL-3.0, see `LICENSE`).
- [AmusementClub/vs-mlrt](https://github.com/AmusementClub/vs-mlrt): TensorRT engine handling, RIFE wrapper logic.
- [emcodem/mvtools](https://github.com/emcodem/mvtools/tree/gpu_work): GPU filters and the token-frame design.
- RIFE: [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE) / [ECCV2022-RIFE](https://github.com/hzwer/ECCV2022-RIFE); NVIDIA TensorRT.

---

# Upstream README (avs-mlrt)

## Description

This project provides AviSynth+ ML filter runtimes for variety of platforms.

This is [a partial port of the VapourSynth plugin vs-mlrt](https://github.com/AmusementClub/vs-mlrt).

To simplify usage, [wrappers](https://github.com/Asd-g/avs-mlrt/tree/main/scripts) are provided for all bundled models.

Custom models can be found [in this doom9 thread](https://forum.doom9.org/showthread.php?t=184768).

### Requirements:

- Vulkan device (`mlrt_ncnn`)

- Intel GPU (`mlrt_ov(... device="GPU")`)

- NVIDIA GPU (`mlrt_ort(... provider="cuda")`)

- DirectX 12-compatible hardware (`mlr_ort(... provider="dml")`)

- AviSynth+ r3928 or later ([1](https://github.com/AviSynth/AviSynthPlus/releases) / [2](https://forum.doom9.org/showthread.php?t=181351) / [3](https://gitlab.com/uvz/AviSynthPlus-Builds))

- Microsoft VisualC++ Redistributable Package 2022 (can be downloaded from [here](https://github.com/abbodi1406/vcredist/releases))

### Filters:

#### mlrt_ncnn

[ncnn](https://github.com/Tencent/ncnn) is a popular AI inference runtime. mlrt_ncnn provides a vulkan based runtime for some AI filters.

It includes support for on-the-fly ONNX to ncnn native format conversion so as to provide a unified interface across all runtimes provided by this project.

[How to use mlrt_ncnn](https://github.com/Asd-g/avs-mlrt/blob/main/mlrt_ncnn/README.md).

#### mlrt_ov

[OpenVINO](https://docs.openvino.ai/latest/index.html) is an AI inference runtime developed by Intel, mainly targeting x86 CPUs and Intel GPUs.

The mlrt_ov plugin provides optimized pure CPU & Intel GPU runtime for some popular AI filters. Intel GPU supports Gen 8+ on Broadwell+ and the Arc series GPUs.

[How to use mlrt_ov](https://github.com/Asd-g/avs-mlrt/blob/main/mlrt_ov/README.md).

#### mlrt_ort

[ONNX Runtime](https://onnxruntime.ai/) is an AI inference runtime with many backends.

The mlrt_ort plugin provides optimized CPU, CUDA GPU and DX12 GPU runtime for some popular AI filters.

[How to use mlrt_ort](https://github.com/Asd-g/avs-mlrt/blob/main/mlrt_ort/README.md).

### Benchmarks

Device specific benchmarks can be found [here](https://github.com/AmusementClub/vs-mlrt/wiki#device-specific-benchmarks).

DirectML benchmarks can be found [here](https://github.com/AmusementClub/vs-mlrt/releases/tag/v13.2).

### Building:

Requirements:
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) (`mlrt_ncnn`)
- [ncnn](https://github.com/Tencent/ncnn) (mlrt_ncnn only)
- [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html) (`mlrt_ov`)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) (`mlrt_ort`)
- [protobuf](https://github.com/protocolbuffers/protobuf)
- [onnx](https://github.com/onnx/onnx)
- [boost](https://www.boost.org/)

```
# Build steps:
# Get the latest Vulkan SDK. (mlrt_ncnn)
# Get the latest ncnn release. (mlrt_ncnn)
# Get the latest OpenVINO release. (mlrt_ov)
# Get the latest Microsoft.ML.OnnxRuntime.DirectML. (mlrt_ort)
# Get the latest CUDA Toolkit. (mlrt_ort)

git clone --recurse-submodules --shallow-submodules --depth 1 https://github.com/Asd-g/avs-mlrt
cd avs-mlrt

# Building protobuf:
cd protobuf
cmake -B build_rel -G Ninja -DCMAKE_BUILD_TYPE=Release -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -DCMAKE_INSTALL_PREFIX=../install
cmake --build build_rel
cmake --install build_rel

# Building onnx:
cd ..\onnx
cmake -B build_rel -G Ninja -DCMAKE_BUILD_TYPE=Release -DONNX_USE_LITE_PROTO=ON -DONNX_USE_PROTOBUF_SHARED_LIBS=OFF -DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_ML=0 -DCMAKE_INSTALL_PREFIX=../install -DONNX_NAMESPACE=ONNX_NAMESPACE -DCMAKE_PREFIX_PATH=../install
cmake --build build_rel
cmake --install build_rel

# Building boost (optional):
#b2 --with-system --with-filesystem --with-chrono -q --toolset=msvc-14.3 address-model=64 variant=release link=static runtime-link=shared threading=multi --hash --prefix=.\bin\x64
#b2 --with-system --with-filesystem --with-chrono -q --toolset=msvc-14.3 address-model=64 variant=release link=static runtime-link=shared threading=multi --hash --prefix=.\bin\x64 install

# Building onnxruntime (mlrt_ort)
cd ..\mlrt_ort\onnxruntime
cmake -B build_rel -DCMAKE_BUILD_TYPE=Release -G Ninja -Donnxruntime_BUILD_SHARED_LIB=ON -Dprotobuf_BUILD_TESTS=OFF -Donnxruntime_BUILD_UNIT_TESTS=OFF -Donnxruntime_USE_CUDA=ON -Donnxruntime_CUDA_HOME="%CUDA_PATH%" -Donnxruntime_CUDNN_HOME=<...> -DCMAKE_INSTALL_PREFIX=..\..\install -DCMAKE_CUDA_ARCHITECTURES="50;61-real;75-real;86-real;89-real" -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -DONNX_USE_PROTOBUF_SHARED_LIBS=OFF cmake -Donnxruntime_USE_DML=ON -DFLATBUFFERS_BUILD_TESTS=OFF cmake
cmake --build build_rel
cmake --install build_rel

# Building avs-mlrt
# BUILD_MRLT_NCNN=ON (default)
# BUILD_MLRT_OV=ON (default)
# BUILD_MLRT_ORT=ON (default)
cd ..\..
cmake -B build -G Ninja -DCMAKE_PREFIX_PATH=".\install";<Vulkan_SDK_path>;<boost_path> -Dncnn_DIR=<...> -DOpenVINO_DIR=<...>
```

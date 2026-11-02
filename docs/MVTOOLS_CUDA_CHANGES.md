# Changes needed in emcodem/mvtools (branch `gpu_work`) for the RIFE/TensorRT plugin

Status: **specification. From here on the RIFE plugin is written as if these changes exist (API v20).**
Reference read: `cudafilters/README.md`, `CudaCommon.h`, `docs/dev/ARCHITECTURE.md`, `Sources/cuda/mvcuda.h` (API v19).

## Why

`mlrt_trt` (RIFE via TensorRT) is limited by PCIe transfers on this machine, not compute (see `optimisation_potential.md`):
17.2 ms compute + 12 ms of fp32 RGB copies that never overlap compute = 33.7 inferences/s. Moving raw 8..16-bit
planar YUV instead of fp32 RGB (conversion on the GPU) cuts the traffic from ~83 MB to ~10 MB per inference, and keeping
frames on the device between filters (token frames) removes the remaining PCIe hops when chained with `CudaUpload` /
`CudaDownload` and other GPU filters. The plugin must therefore read and write token frames.

## What v19 cannot do

`mvcuda.h` v19 has upload (host to device), download (device to host), pin/unpin, producer register/regenerate and stats.
There is **no way for an external plugin to**
1. get the device pointers of a token's frame (to run its own kernels / TensorRT on it), or
2. create a registry frame backed by device memory it writes itself (to return a token).

The only workaround today is `mvcuda_frame_download` + our own upload, which defeats the purpose.

## Required additions (API v20, `MVCUDA_API_VERSION` 19 to 20)

Design rule kept: no CUDA types in the public header (device addresses are `unsigned long long`, streams are `void *`
that the caller casts from its `cudaStream_t`). The RIFE plugin brings its own conversion kernels and TensorRT; mvtools_cuda
only has to hand out device memory and keep the ordering/lifetime right. (Alternative considered and rejected for now:
putting `pack`/`unpack` YUV<->RGB-fp16 kernels into mvtools_cuda. It works too, but then standalone use without
mvtools_cuda needs a second copy of the kernels anyway.)

```c
// ---- interop: external consumers/producers of device frames (API 20) -------------------------------------------

// Device view of the frame of a token. planes[p] = device address of plane p (0 Y, 1 U, 2 V; unused = 0),
// pitch_bytes[p] = row pitch in bytes (>= plane width * bytes_per_sample, the library's allocation pitch).
// The call makes `wait_stream` (a cudaStream_t, may be 0 = legacy default stream) wait for the work that produced the
// frame (the registry's ready marks / DevFrame::seq), and holds the frame (as mvcuda_token_pin does) until the matching
// release. Does not block the host.
// Returns 0, MVCUDA_TOKEN_GONE (-6) when evicted (caller regenerates as for any token), -4 bad arguments.
MVCUDA_API int mvcuda_token_acquire(unsigned long long token, MvCudaFrameFormat *format,
  unsigned long long planes[3], int pitch_bytes[3], void *wait_stream);

// Ends the hold. `done_stream`: the stream on which the caller's reads were enqueued; the library records an event there
// and does not reuse/free the buffer before it completes (so the registry may evict at once, memory is returned later).
MVCUDA_API void mvcuda_token_release(unsigned long long token, void *done_stream);

// Allocates a new registry frame of `format` (pooled like every other frame, counted in the byte budget, may evict others)
// for the caller to write on `stream`. out receives the header to put in the token frame; the token is usable by consumers
// only after mvcuda_frame_commit. producer/frame: the regeneration source (mvcuda_producer_register id of the caller).
MVCUDA_API int mvcuda_frame_alloc(const MvCudaFrameFormat *format, unsigned long long producer, int frame,
  unsigned long long planes[3], int pitch_bytes[3], MvCudaTokenHeader *out);

// Marks the frame ready: records an event on `stream` after the caller's writes. Consumers (acquire / the library's own
// filters / mvcuda_frame_download) wait on that event, never on the host. If the caller aborts instead, mvcuda_frame_discard.
MVCUDA_API int mvcuda_frame_commit(unsigned long long token, void *stream);
MVCUDA_API void mvcuda_frame_discard(unsigned long long token);
```

Semantics that matter:
- **Ordering is stream/event based, never a host wait**, same as the library's own submit model (`DevFrame::seq` + marks).
  `mvcuda_frame_download` on a committed frame must wait for its event. (The plugin may host-sync its own stream when it
  needs to reuse its engine buffers; that is its business.)
- **Thread safety:** all five calls callable from many Prefetch threads concurrently, like `mvcuda_frame_upload_keyed`.
- **Different CUDA runtimes:** mvtools_cuda links cudart 12.x statically; the plugin uses cudart 13 (TensorRT 10.16). Both
  use the device's primary context, so device addresses and `CUstream` handles are interchangeable. **To be verified by a
  test** (acquire a token, read it from a cudart-13 kernel on a cudart-13 stream, compare with `mvcuda_frame_download`).
  If this does not hold, the fallback is building mvtools_cuda against CUDA 13 (sm_120 is supported from 12.8 on).
- **Formats:** the existing `MvCudaFrameFormat` restrictions apply (planar Y or YUV, 8..16 bit integer, subsampling 1 or 2).
  RIFE needs YUV 4:2:0 / 4:2:2 / 4:4:4 at 8, 10 (and up to 16) bit. Frames whose format the plugin does not support:
  the plugin downloads them (`mvcuda_frame_download`) and takes the host path.
- **Regeneration:** the plugin registers a producer (`mvcuda_producer_register`) whose function recomputes frame n (calls
  its own `GetFrame(n)` path and returns the fresh token). Evicted tokens it produced are therefore recoverable, as for every
  other GPU filter.
- **Budget accounting:** frames from `mvcuda_frame_alloc` count in the registry budget exactly like uploaded/produced
  ones. Engine input/output buffers of the plugin are NOT registry frames (the plugin owns and sizes them).
- **Keyed upload** (`mvcuda_frame_upload_keyed`) stays the way CPU frames enter: each source picture is uploaded once even
  though RIFE reads it for ~3 different (frame n, n+1) pairs at 30 to 50 fps.
- `mvcuda_api_version()` returns 20; the plugin refuses another version and then runs without mvtools_cuda (standalone path).

## Plugin behaviour (mlrt_trt) once these exist

Modes, chosen per call from what arrives (nothing for the user to set except `cudadownload`):

| Input frames | mvtools_cuda present | Path | Output |
|---|---|---|---|
| token frames (YUV) | yes | acquire tokens, GPU YUV to RGB-fp16, TensorRT, GPU RGB to YUV, alloc+commit | token frame if `cudadownload=false`, else downloaded CPU frame |
| CPU YUV frames (8..16 bit planar) | yes | `mvcuda_frame_upload_keyed` then as above | same |
| CPU YUV frames | no (standalone) | plugin's own pinned raw-YUV upload + own kernels | CPU frame (always) |
| CPU RGB fp32 planar (today's interface) | irrelevant | unchanged legacy path (zimg conversions stay in the script) | CPU frame |

- **`cudadownload` (bool, default `true`)**: for the YUV paths with mvtools_cuda. `true`: the filter returns ordinary
  frames in the input's pixel format (a device frame is downloaded into the result), so it drops into any script and into
  standalone use unchanged. `false`: returns token frames (header in luma row 0 and every 8th luma row black, as the other
  GPU filters do), to be consumed by other GPU filters or by an explicit `CudaDownload`. Ignored (always CPU out) on the
  standalone and legacy RGB paths. Default true keeps "normal frames in, normal frames out" from the library's usage rule;
  a token frame in with `cudadownload=true` still returns normal frames.
- **Timestep** is read as the first sample of the existing `tp` clip (a CPU Y32 frame; only 4 bytes are used, the plane is
  never uploaded) and written by the plugin's kernel on the device. This also removes the 8 MB constant plane.
- **Padding** to the engine's tile (1080 to 1088): done by the plugin's pack kernel. Default: black like today's
  `AddBorders`, so output stays comparable to the current pipeline; option `pad=replicate` later.
- **Standalone** must keep working with only `mlrt_trt.dll` + its TensorRT/CUDA DLLs (no mvtools2 / mvtools_cuda / cudafilters).
  The mvtools_cuda library is loaded dynamically from the plugin's folder (or next to cudafilters.dll), like the other
  plugins do; missing library, no device, or API mismatch = standalone path, silently.
- Colour handling in the kernels: BT.709 limited range first, Catmull-Rom (bicubic b=0,c=0.5, zimg's default) chroma
  upsampling and the matching downsampling; BT.601 / BT.2020 ncl and full range as parameters afterwards. Output verified
  against the current zimg path (PSNR/SSIM plus visual check on real footage) before it becomes the default.

## Validation (mirrors the library's own rules)

1. framemd5 of the token path (`cudadownload=false` then `CudaDownload`) vs `cudadownload=true` vs the standalone path:
   must be identical (same kernels, same engine).
2. Tiny registry budgets (`MVCUDA_CACHE_MB=1..16`) must give the same md5 (regeneration path).
3. Negative controls (a deliberately wrong matrix) must differ.
4. Cross-runtime test above (cudart 13 reading a token produced by cudart 12.x code).
5. Speed: `ffmpeg -frames:v 1500 -f null -`, same clip, compare inferences/s against the 33.7 / 42.3 qps baselines.

---

## Status (2026-11-02): plugin side implemented, tested against a mock only

mlrt_trt implements the consumer/producer side against the v20 API above (`src/mvcuda_interop.h`, `run_mv()` /
`mv_regenerate()` / `get_frame_mlrt_trt_mv()` in `src/mlrt_trt.cpp`). It is only active when `mvtools_cuda.dll` is loaded (or
found next to the plugin), reports API version 20 and has a device; otherwise the standalone YUV path runs, silently.
Parameter `mvtools_cuda` (default true) switches the interop off.

Calls the plugin makes, per RIFE call: for each of the 2 input frames `mvcuda_frame_upload_keyed` (CPU frame only; key = clip
identity x frame number) -> `mvcuda_token_acquire(token, ..., wait_stream = instance stream)` -> kernels + TensorRT on that
stream -> `mvcuda_token_release(token, same stream)`. Output with `cudadownload=false`: `mvcuda_frame_alloc` -> unpack kernel
writes the frame (pitched) -> `mvcuda_frame_commit(token, stream)`, then the token header is written into luma row 0 and every
8th luma row is blacked. Evicted input tokens: `mvcuda_token_regenerate(producer, frame)` and one retry. Output frames can be
regenerated through the producer registered with `mvcuda_producer_register`. No host wait in the token-out path: the next call
on the same instance queues behind it on the instance's stream.

Tested with `tools/mock_mvtools_cuda.cpp` (a mock library implementing the 5 new functions + upload/download with plain
cudaMallocPitch frames, plus MockUpload / MockDownload filters): CPU in, token in -> token out, token in -> CPU out are all
**bit-identical** to the standalone path over 300 frames. NOT tested: the real library's event ordering, eviction/regeneration
under a small budget, budget accounting, a cudart-12-vs-13 context mix. Those need the real v20 implementation (see Validation).
Assumptions made: U and V planes of a device frame share one pitch (checked, error otherwise); the registry hands out
pointers valid until `mvcuda_token_release` has been ordered on `done_stream`.

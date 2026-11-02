### FAST PATH: 29.97fps -> 50fps RIFE interpolation straight from the source's YUV (YV12 / YUV422P10 / any planar 8..16 bit). ###
### The GPU does YUV->RGB, padding to the engine tile and the crop back; only raw YUV crosses PCIe. Measured ~1.28x realtime ###
### on the RTX 5050 test machine (the RGB path in example_50p.avs: ~0.84x), output within engine-noise of the RGB path.      ###
### Needs an fp16-I/O engine (models\rife_v2\rife_v4.17_lite_fp16io.engine) or pass onnx_path= to build one on demand.       ###
### Encode with ffmpeg's fps filter for true 50.000fps CFR:  ffmpeg -i example_50p_yuv.avs -vf fps=50 -c:v <codec> out.mp4   ###

here = ScriptDir()

LoadPlugin(here + "\mlrt_trt.dll")
LoadPlugin("C:\dev\avs_rife\avisynth\plugins\ffms2.dll")
Import(here + "\scripts\mlrt_RIFE_trt.avsi")

# --- Replace with your actual source clip (8..16 bit planar YUV; no conversion needed) ---
c = FFVideoSource("D:\kuerbiskern_orig.MP4")

# relabel 29.97 as 30 before the retiming (3:5 ratio, 20% of output frames are exact source frames and skip RIFE)
c = AssumeFPS(c, 30, 1)

# defaults: matrix="709", range="limited" (set matrix="601"/"2020", range="full" if your source differs),
# chroma sited left/centred (MPEG-2/H.264). num_streams=2 is enough (the GPU runs one inference at a time anyway).
c = mlrt_RIFE_trt(c, here + "\models\rife_v2\rife_v4.17_lite_fp16io.engine", 2, fps_num=50, fps_den=1, device=0, num_streams=2, use_cuda_graph=true)

# duration-correct relabel; the ffmpeg fps filter does the real CFR conversion (see example_50p.avs)
c = AssumeFPS(c, 50000, 1001)
Prefetch(c, 8, 16)

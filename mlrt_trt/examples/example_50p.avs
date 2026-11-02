### Example: 29.97fps -> 50fps RIFE interpolation, the current fastest verified-clean, ###
### no-quality-cost config (validated 2026-08-13, see README.md's "Current fastest       ###
### verified-clean, no-quality-cost config" and "Duration fix" sections for the full     ###
### derivation and benchmark numbers). Measured ~0.868-0.871x realtime on the machine     ###
### this package was built on (RTX 5050, 8GB) -- re-tune Prefetch/num_streams if your     ###
### target hardware differs significantly (see PORTING_TO_AVISYNTH.md).                  ###
###                                                                                        ###
### All paths below are relative to this script's own folder (ScriptDir()) EXCEPT the     ###
### source clip and the generic AviSynth+ plugins (ffms2/avsresize), which are assumed    ###
### already present on the host from a normal AviSynth+ install -- this package itself    ###
### only bundles the mlrt_trt plugin, its CUDA/TensorRT runtime, and the RIFE models.      ###
###                                                                                        ###
### IMPORTANT: encode this script with ffmpeg's `fps` filter to get true 50.000fps CFR    ###
### output with the correct total duration -- e.g.:                                       ###
###   ffmpeg -i example_50p.avs -vf fps=50 -c:v <your codec> output.mp4                   ###
### Do NOT just use -r 50 with no filter / skip the fps filter -- without it the container ###
### will be tagged 49.950fps (AssumeFPS below is a duration-correct relabel, not the final ###
### delivery rate). See README.md's "Duration fix" section for why.                        ###

here = ScriptDir()

LoadPlugin(here + "\mlrt_trt.dll")
LoadPlugin("C:\dev\avs_rife\avisynth\plugins\ffms2.dll")
LoadPlugin("C:\dev\avs_rife\avisynth\plugins\avsresize.dll")
Import(here + "\scripts\mlrt_RIFE_trt.avsi")

# --- Replace with your actual source clip ---
c = FFVideoSource("C:\Temp\kuerbiskern_orig.MP4")

# Relabel the true 29.97fps source as exactly 30fps before the RIFE retiming math:
# reduces the 30:50 ratio to 3:5 (20% of output frames land exactly on a source frame and
# skip RIFE entirely), vs. ~0.1% for the true 600:1001 ratio. This compresses the timeline
# by 1000/1001 -- corrected below via the AssumeFPS(50000,1001) relabel after interpolation.
c = AssumeFPS(c, 30, 1)

# YUV -> 32-bit planar float RGB. Adjust colorspace_op if your source's color tags differ.
c = z_ConvertFormat(c, pixel_type="RGBPS", colorspace_op="709:709:709:l=>rgb:709:709:f")

# pad 1080 -> 1088: implementation=2 pads internally and doesn't strictly require this, but
# the validated 0.868-0.871x benchmark was measured WITH this padding (marginally faster than
# native 1920x1080 in this project's own testing) -- keep it for parity with that number.
# Adjust the border amount if your source resolution differs (must round up to a multiple of 32).
c = AddBorders(c, 0, 0, 0, 8)

c = mlrt_RIFE_trt(c, here + "\models\rife_v2\rife_v4.17_lite_fp16.engine", 2,
    \ fps_num=50, fps_den=1, device=0, num_streams=2, use_cuda_graph=true)

# Relabel at the duration-correct rate (metadata only, no resample) -- see the ffmpeg
# `fps` filter note above; this alone is NOT sufficient to get a literal 50.000fps output.
c = AssumeFPS(c, 50000, 1001)

c = Crop(c, 0, 0, 1920, 1080)
c = z_ConvertFormat(c, pixel_type="YV12", colorspace_op="rgb:709:709:f=>709:709:709:l")
Prefetch(c, 8, 16)

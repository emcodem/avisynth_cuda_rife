### A mlrt_trt (TensorRT backend) wrapper of RIFE, mirroring mlrt_RIFE.avsi's frame-interleaving
### logic but calling the low-level mlrt_trt(...) filter, which supports two modes:
###   - engine_path only: loads a pre-built .engine file as-is (fastest, requires the exact frame
###     size the engine was built for -- use this when the source resolution is known ahead of time).
###   - onnx_path (with engine_path omitted, or "", or pointing at a file that doesn't exist yet):
###     builds (and caches to disk, named by resolution/GPU/TensorRT version) an engine from the
###     ONNX network on first use via trtexec.exe, then reuses the cached file on every later call
###     at that same resolution/hardware. Mirrors vsmlrt.py's own get_engine_path()/trtexec()
###     build+cache logic -- use this when source resolution varies and isn't known ahead of time.
###     Needs trtexec.exe next to mlrt_trt.dll (or passed via trtexec_path).
###
### This is a port of https://github.com/AmusementClub/vs-mlrt/blob/master/scripts/vsmlrt.py's
### RIFE()/RIFEMerge() for the TensorRT backend, combined with mlrt_RIFE.avsi's implementation-1/2
### input-plane construction.


### Requirements
# mlrt_trt


### Usage ###
###
# Function mlrt_RIFE_trt(clip c, string engine_path, int implementation, int "factor_num", int "factor_den", int "fps_num", int "fps_den", int "overlap_w", int "overlap_h", int "tilesize_w", int "tilesize_h", int "device", bool "use_cuda_graph", int "num_streams", int "verbosity", bool "sc", float "sc_threshold", string "onnx_path", string "trtexec_path")
###
## Parameters ##
#---------------
# c: Input clip. Must be in RGB 32-bit planar format.
#---------------
# engine_path: Path to a pre-built TensorRT engine file. Pass "" (or omit a resolution-specific
# path that doesn't exist yet) together with onnx_path to build-and-cache one on demand instead.
#---------------
# onnx_path: Path to the RIFE ONNX network (e.g. ".../rife_v4.17.onnx", implementation-1 graph).
# Only used when engine_path is "" or doesn't exist yet -- builds an engine via trtexec.exe sized
# exactly to c's (post-padding) width/height, caches it to disk, and reuses it on later calls at
# the same resolution/GPU/TensorRT version. First call at a new resolution costs ~30-90s to build.
#---------------
# trtexec_path: Path to trtexec.exe. Default: looked up next to mlrt_trt.dll. Only relevant when
# onnx_path triggers a build.
#---------------
# implementation: Which RIFE onnx graph the engine was built from. Must be 1 or 2.
# 1: positional-encoding inputs (clipa, clipb, timepoint, horizontal, vertical, multiplier_h, multiplier_v) -- 11 input planes.
#    Video width/height must be mod32 (tilesize_requirement=32 for RIFE v4.x < v4.25; see vsmlrt.py's RIFEMerge for exceptions).
# 2: internal padding (clipa, clipb, timepoint) -- 7 input planes, no mod32 requirement.
#---------------
# factor_num, factor_den: Factor of target frame rate. Default: factor_num = 2, factor_den = 1.
# If factor_num/factor_den is not an exact integer, this becomes a general rational
# rate conversion (see fps_num/fps_den below) -- this includes decreasing the frame rate.
#---------------
# fps_num, fps_den: Target frame rate. If specified, supersedes factor_num/factor_den.
# Any rational ratio is supported, including a DECREASE from the source rate
# (e.g. 29.97 -> 25): when the ratio isn't an exact integer multiple, each output frame
# is produced by interpolating between its two nearest source frames at the correct
# timepoint, rather than by the fixed even-interleave grid used for integer multiples.
#---------------
# overlap_w, overlap_h, tilesize_w, tilesize_h: passed through to mlrt_trt. Default: no tiling (full frame).
#---------------
# device, use_cuda_graph, num_streams, verbosity: passed through to mlrt_trt.
#---------------
# sc, sc_threshold: as in mlrt_RIFE.avsi -- avoid interpolating over scene changes.
#---------------
# YUV MODE (fast path): pass a planar integer YUV clip (8..16 bit, 4:2:0 / 4:2:2 / 4:4:4, e.g. YV12 / YUV422P10) instead of
# fp32 planar RGB. The plugin then moves only the raw YUV planes over PCIe and does YUV->RGB, the padding to the engine tile
# and the crop back ON THE GPU (no z_ConvertFormat / AddBorders / Crop in the script, no fp32 RGB frames). Needs
# implementation=2 and an engine with fp16 I/O (onnx_path builds one automatically; for a pre-built engine use
# trtexec --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw). The output clip has the input's pixel type and size.
# matrix ("709" default, "601", "2020"), range ("limited" default, "full"), pad_multiple (32 default),
# cudadownload (default true): passed through to mlrt_trt. In YUV mode sc_threshold is in 8-bit luma units (0..255).


Function mlrt_RIFE_trt(clip c, string engine_path, int implementation, int "factor_num", int "factor_den", int "fps_num", int "fps_den", int "overlap_w", int "overlap_h", int "tilesize_w", int "tilesize_h", int "device", bool "use_cuda_graph", int "num_streams", int "verbosity", bool "sc", float "sc_threshold", string "onnx_path", string "trtexec_path", string "matrix", string "range", int "pad_multiple", bool "cudadownload")
{
    factor_num = Default(factor_num, 2)
    factor_den = Default(factor_den, 1)
    sc = Default(sc, False)
    sc_threshold = Default(sc_threshold, 0.1)
    onnx_path = Default(onnx_path, "")
    trtexec_path = Default(trtexec_path, "")

    yuv_mode = IsYUV(c) && IsPlanar(c) && NumComponents(c) == 3 && BitsPerComponent(c) <= 16
    Assert(yuv_mode || (ComponentSize(c) == 4 && IsPlanarRGB(c)), "mlrt_RIFE_trt: clip must be 32-bit planar RGB (legacy path) or planar integer YUV, 8..16 bit (fast GPU path).")
    Assert(!yuv_mode || implementation == 2, "mlrt_RIFE_trt: YUV mode requires implementation=2.")
    Assert(implementation == 1 || implementation == 2, "mlrt_RIFE_trt: implementation must be either 1 or 2.")
    Assert(engine_path != "" || onnx_path != "", "mlrt_RIFE_trt: either engine_path or onnx_path must be given.")

    trt_args = ""
    trt_args = Defined(overlap_w) ? trt_args + ", overlap_w=" + String(overlap_w) : trt_args
    trt_args = Defined(overlap_h) ? trt_args + ", overlap_h=" + String(overlap_h) : trt_args
    trt_args = Defined(tilesize_w) ? trt_args + ", tilesize_w=" + String(tilesize_w) : trt_args
    trt_args = Defined(tilesize_h) ? trt_args + ", tilesize_h=" + String(tilesize_h) : trt_args
    trt_args = Defined(device) ? trt_args + ", device=" + String(device) : trt_args
    trt_args = Defined(use_cuda_graph) ? trt_args + ", use_cuda_graph=" + String(use_cuda_graph) : trt_args
    trt_args = Defined(num_streams) ? trt_args + ", num_streams=" + String(num_streams) : trt_args
    trt_args = Defined(verbosity) ? trt_args + ", verbosity=" + String(verbosity) : trt_args
    trt_args = (yuv_mode && Defined(matrix)) ? trt_args + ", matrix=" + Chr(34) + matrix + Chr(34) : trt_args
    trt_args = (yuv_mode && Defined(range)) ? trt_args + ", range=" + Chr(34) + range + Chr(34) : trt_args
    trt_args = (yuv_mode && Defined(pad_multiple)) ? trt_args + ", pad_multiple=" + String(pad_multiple) : trt_args
    trt_args = (yuv_mode && Defined(cudadownload)) ? trt_args + ", cudadownload=" + String(cudadownload) : trt_args

    multi = (Defined(fps_num) && Defined(fps_den)) ?
        \ ((Float(fps_num) / fps_den) / FrameRate(c)) :
        \ (Float(factor_num) / factor_den)

    c = yuv_mode ? c : Expr(c, "x 0 1 clip")

    if (Frac(multi) == 0.0)
    {
        multi = Int(multi)
        Assert(multi >= 2, "mlrt_RIFE_trt: an exact-integer frame rate multiple must be at least 2x the source frame rate; a non-integer fps_num/fps_den ratio (e.g. downsampling) goes through the general path below instead.")

        c_array = [c]
        multi_minus_one = multi - 1
        for (i = 2, multi_minus_one)
        {
            c_array = ArrayAdd(c_array, c)
        }
        initial = Interleave(c_array)

        terminal = DuplicateFrame(c, FrameCount(c) - 1)
        terminal = Trim(terminal, 1, 0)
        terminal_array = [terminal]
        for (i = 2, multi_minus_one)
        {
            terminal_array = ArrayAdd(terminal_array, terminal)
        }
        terminal = Interleave(terminal_array)

        timepoint_array = []
        for (i = 1, multi_minus_one)
        {
            timepoint_array = ArrayAdd(timepoint_array, BlankClip(clip=c, colors=[Float(i) / multi], pixel_type="y32"))
        }
        timepoint = Interleave(timepoint_array)

        output0 = RIFEMerge_trt(initial, terminal, timepoint, implementation, engine_path, onnx_path, trtexec_path, trt_args)

        if (sc)
        {
            y_clip = yuv_mode ? output0 : z_ConvertFormat(output0, pixel_type="y32", colorspace_op="rgb:709:709:f=>709:709:709:l")
            sc_threshold = yuv_mode ? sc_threshold * 255.0 : sc_threshold
            output0 = ScriptClip(output0, function [y_clip, initial, sc_threshold] ()
            {
                if (YDifferenceToNext(y_clip) > sc_threshold)
                {
                    return Trim(initial, current_frame, current_frame)
                }
            })
        }

        if (multi == 2)
        {
            res = Interleave(c, output0)
        }
        else
        {
            res_array = [c]
            for (i = 0, multi_minus_one - 1)
            {
                res_array = ArrayAdd(res_array, SelectEvery(output0, multi - 1, i))
            }
            res = Interleave(res_array)
        }

        return (FrameRateNumerator(c) != 0 && FrameRateDenominator(c) != 0) ?
            \ AssumeFPS(res, FrameRateNumerator(c) * multi, FrameRateDenominator(c))
            \ : res
    }
    else
    {
        # General rational fps conversion (any ratio, including a rate DECREASE like
        # 29.97->25): for each output frame, pick its two bracketing source frames and
        # the exact interpolation timepoint between them, instead of the fixed 1/multi
        # grid the integer-multiple path above uses. Mirrors vsmlrt.py RIFE()'s
        # fractional-multi branch, using ScriptClip + Trim in place of VapourSynth's
        # FrameEval/PickFrames.
        src_fps_num = FrameRateNumerator(c)
        src_fps_den = FrameRateDenominator(c)
        src_fps = (src_fps_num == 0 || src_fps_den == 0) ? 1.0 : FrameRate(c)
        dst_fps = src_fps * multi
        src_frames = FrameCount(c)
        dst_frames = Min(Int(src_frames * multi), Int(Pow(2, 31)) - 1)

        Assert(dst_frames >= 1, "mlrt_RIFE_trt: target frame rate produces zero output frames.")

        duration_rel = src_fps / dst_fps
        dst_duration = ContinuedNumerator(duration_rel, limit=50000)
        src_duration = ContinuedDenominator(duration_rel, limit=50000)

        temp = BlankClip(clip=c, length=dst_frames)

        left_clip = ScriptClip(temp, function [c, src_duration, dst_duration, src_frames] ()
        {
            idx = Min(dst_duration * current_frame / src_duration, src_frames - 1)
            return Trim(c, idx, idx)
        })

        right_clip = ScriptClip(temp, function [c, src_duration, dst_duration, src_frames] ()
        {
            idx = Min(dst_duration * current_frame / src_duration + 1, src_frames - 1)
            return Trim(c, idx, idx)
        })

        temp_gray = BlankClip(clip=temp, pixel_type="y32")
        tp_clip = ScriptClip(temp_gray, function [src_duration, dst_duration] ()
        {
            current_time = dst_duration * current_frame
            return BlankClip(clip=last, colors=Float(current_time - src_duration * (current_time / src_duration)) / src_duration)
        })

        output0 = RIFEMerge_trt(left_clip, right_clip, tp_clip, implementation, engine_path, onnx_path, trtexec_path, trt_args)

        y_clip = yuv_mode ? output0 : z_ConvertFormat(output0, pixel_type="y32", colorspace_op="rgb:709:709:f=>709:709:709:l")
        sc_threshold = yuv_mode ? sc_threshold * 255.0 : sc_threshold
        res = ScriptClip(output0, function [left_clip, src_duration, dst_duration, src_frames, sc, y_clip, sc_threshold] ()
        {
            current_time = dst_duration * current_frame
            lands_on_source_frame = (current_time % src_duration == 0) || ((current_time / src_duration) + 1 >= src_frames)
            if (lands_on_source_frame || (sc && YDifferenceToNext(y_clip) > sc_threshold))
            {
                return left_clip
            }
        })

        return (src_fps_num != 0 && src_fps_den != 0) ?
            \ AssumeFPS(res, ContinuedNumerator(dst_fps, limit=50000), ContinuedDenominator(dst_fps, limit=50000))
            \ : res
    }
}


Function RIFEMerge_trt(clip clipa, clip clipb, clip clipc, int implementation, string engine_path, string onnx_path, string trtexec_path, string trt_args)
{
    if (implementation == 1)
    {
        Assert(!(Width(clipa) % 32 != 0 || Height(clipa) % 32 != 0), "mlrt_RIFE_trt: width/height must be divisible by 32 (" + String(Width(clipa)) + ", " + String(Height(clipa)) + ").")

        empty = BlankClip(clip=clipa, pixel_type="y32")
        horizontal = Expr(empty, "sx 2 * width 1 - / 1 -")
        vertical = Expr(empty, "sy 2 * height 1 - / 1 -")

        multiplier_h = BlankClip(clip=clipa, pixel_type="y32", colors=2.0/(Width(clipa) - 1))
        multiplier_v = BlankClip(clip=clipa, pixel_type="y32", colors=2.0/(Height(clipa) - 1))

        return Eval("mlrt_trt([clipa, clipb, clipc, horizontal, vertical, multiplier_h, multiplier_v], engine_path=engine_path, onnx_path=onnx_path, trtexec_path=trtexec_path" + trt_args + ")")
    }
    else
    {
        return Eval("mlrt_trt([clipa, clipb, clipc], engine_path=engine_path, onnx_path=onnx_path, trtexec_path=trtexec_path" + trt_args + ")")
    }
}

hdr = r'''here = ScriptDir()
LoadPlugin(here + "\..\mlrt_trt.dll")
LoadPlugin("C:\dev\avs_rife\avisynth\plugins\ffms2.dll")
Import(here + "\..\scripts\mlrt_RIFE_trt.avsi")
c = FFVideoSource("D:\kuerbiskern_orig.MP4")
c = AssumeFPS(c, 30, 1)
'''
tail = r'''c = mlrt_RIFE_trt(c, here + "\..\rife_v4.17_lite_fp16io.engine", 2, fps_num=50, fps_den=1, device=0, num_streams=2, use_cuda_graph=true)
'''
tests = {
    'p10_422': 'c = ConvertToYUV422(c)\nc = ConvertBits(c, 10)\n' + tail + 'c = ConvertBits(c, 8, dither=-1)\nc = ConvertToYV12(c)\nPrefetch(c,8,16)\n',
    'yuv444': 'c = ConvertToYUV444(c)\n' + tail + 'c = ConvertToYV12(c)\nPrefetch(c,8,16)\n',
    'odd_size': 'c = Crop(c, 2, 2, 1916, 1076)\n' + tail + 'Prefetch(c,8,16)\n',
    'sc_on': tail.replace('use_cuda_graph=true)', 'use_cuda_graph=true, sc=true, sc_threshold=0.1)') + 'Prefetch(c,8,16)\n',
}
for k, v in tests.items():
    open(f'fmt/{k}.avs', 'w').write(hdr + v)
print(list(tests))

import shutil
shutil.copyfile(r'D:\dev\avs_rife_tensor\deploy\mlrt_trt\cudart64_13.dll', r'D:\dev\avs_rife_tensor\test_yuv\mock\cudart64_13.dll')
common = r'''LoadPlugin("D:\dev\avs_rife_tensor\test_yuv\mlrt_trt.dll")
LoadPlugin("C:\dev\avs_rife\avisynth\plugins\ffms2.dll")
Import("D:\dev\avs_rife_tensor\test_yuv\scripts\mlrt_RIFE_trt.avsi")
c = FFVideoSource("D:\kuerbiskern_orig.MP4")
c = AssumeFPS(c, 30, 1)
'''
mock = r'''LoadPlugin("D:\dev\avs_rife_tensor\test_yuv\mock\mvtools_cuda.dll")
'''
eng = r'''"D:\dev\avs_rife_tensor\test_yuv\rife_v4.17_lite_fp16io.engine"'''
call = lambda extra: 'c = mlrt_RIFE_trt(c, ' + eng + ', 2, fps_num=50, fps_den=1, device=0, num_streams=2, use_cuda_graph=true' + extra + ')\n'
tail = 'c = AssumeFPS(c, 50000, 1001)\nPrefetch(c, 8, 16)\n'
scripts = {
    # baseline: no mvtools_cuda anywhere -> standalone YUV path
    'mv_base': common + call('') + tail,
    # CPU frames in, cudadownload default (true): library upload (keyed) + our own raw-YUV download
    'mv_cpu_in': mock + common + call('') + tail,
    # token frames in AND out (cudadownload=false), consumed by MockDownload
    'mv_tokens': mock + common + 'c = MockUpload(c)\n' + call(', cudadownload=false') + 'c = MockDownload(c)\n' + tail,
    # token frames in, normal frames out (cudadownload=true)
    'mv_tok_in_cpu_out': mock + common + 'c = MockUpload(c)\n' + call(', cudadownload=true') + 'c = MockDownload(c)\n' + tail,
}
for k, v in scripts.items():
    open(rf'D:\dev\avs_rife_tensor\test_yuv\{k}.avs', 'w').write(v)
print(list(scripts))

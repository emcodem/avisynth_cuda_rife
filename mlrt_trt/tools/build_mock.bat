@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set R=D:\dev\avs_rife_tensor
if not exist %R%\test_yuv\mock mkdir %R%\test_yuv\mock
cl /nologo /LD /MT /EHsc /O2 /std:c++20 /I%R%\deps\avisynth\include /I%R%\deps\cuda\include /I%R%\deps\cuda\include\targets\x64 /I%R%\deps\cuda\include\targets\x64\cccl mock_mvtools_cuda.cpp /Fe%R%\test_yuv\mock\mvtools_cuda.dll /Fo%R%\test_yuv\mock\ /link /LIBPATH:%R%\deps\cuda\lib cudart.lib %R%\deps\avisynth\lib\avisynth.lib delayimp.lib /DELAYLOAD:AviSynth.dll

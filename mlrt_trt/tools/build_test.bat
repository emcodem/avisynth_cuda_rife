@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set CE=D:\dev\avs_rife_tensor\_cuda_headers_env\Library
"%CE%\bin\nvcc.exe" -arch=sm_120 -O2 -cudart shared -I"%CE%\include" -L"%CE%\lib\x64" -o test_rife_io.exe test_rife_io.cu ..\src\rife_io.cu

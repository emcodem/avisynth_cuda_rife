@echo off
rem Builds mlrt_trt.dll (Release, Ninja). Needs Visual Studio 2022 (vcvars64) and a CUDA nvcc.
rem Dependencies are looked up under DEPS_ROOT (default D:\dev\avs_rife_tensor) in the layout used so far:
rem   %DEPS_ROOT%\deps\avisynth\{include,lib\avisynth.lib}   AviSynth+ C API headers + import lib
rem   %DEPS_ROOT%\deps\tensorrt                              TensorRT SDK (10.x)
rem   %DEPS_ROOT%\deps\cuda\{include,lib\cudart.lib}         CUDA runtime headers / import lib
rem   %DEPS_ROOT%\_cuda_headers_env\Library                  conda env providing nvcc and the full CUDA include dir
rem Override with:  set DEPS_ROOT=...   set NVCC=...   set NVCC_INCLUDE=...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if "%DEPS_ROOT%"=="" set DEPS_ROOT=D:\dev\avs_rife_tensor
if "%NVCC%"=="" set NVCC=%DEPS_ROOT%\_cuda_headers_env\Library\bin\nvcc.exe
if "%NVCC_INCLUDE%"=="" set NVCC_INCLUDE=%DEPS_ROOT%\_cuda_headers_env\Library\include
set CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set SRC=%~dp0..
"%CMAKE%" -S "%SRC%" -B "%SRC%\build" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DAVISYNTH_INCLUDE_DIRECTORY="%DEPS_ROOT%\deps\avisynth\include" ^
  -DAVISYNTH_LIBRARY="%DEPS_ROOT%\deps\avisynth\lib\avisynth.lib" ^
  -DTENSORRT_HOME="%DEPS_ROOT%\deps\tensorrt" ^
  -DCUDA_INCLUDE_DIRECTORY="%DEPS_ROOT%\deps\cuda\include" ^
  -DCUDA_LIBRARY_DIRECTORY="%DEPS_ROOT%\deps\cuda\lib" ^
  -DNVCC="%NVCC%" -DNVCC_INCLUDE_DIRECTORY="%NVCC_INCLUDE%" || exit /b 1
"%CMAKE%" --build "%SRC%\build" || exit /b 1

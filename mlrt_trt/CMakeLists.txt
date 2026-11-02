cmake_minimum_required(VERSION 3.28)

project(mlrt_trt LANGUAGES CXX)

set(AVISYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to AviSynth+ C API headers")
set(AVISYNTH_LIBRARY "" CACHE FILEPATH "Path to avisynth.lib (import lib for AviSynth.dll's C API)")
set(TENSORRT_HOME "" CACHE PATH "Path to TensorRT SDK")
set(CUDA_INCLUDE_DIRECTORY "" CACHE PATH "Path to CUDA runtime headers (cuda_runtime.h)")

# The GPU YUV<->RGB-fp16 stages (src/rife_io.cu) are compiled with nvcc into an object that is linked into the plugin
# (no nvcc runtime dependency beyond cudart, which the plugin already uses).
set(NVCC "" CACHE FILEPATH "Path to nvcc.exe (needs cuda_fp16.h next to the CUDA headers it is given)")
set(NVCC_INCLUDE_DIRECTORY "" CACHE PATH "Full CUDA include directory for nvcc (cuda_fp16.h, cuda_runtime.h, ...)")
set(RIFE_IO_OBJ ${CMAKE_CURRENT_BINARY_DIR}/rife_io.obj)
add_custom_command(
    OUTPUT ${RIFE_IO_OBJ}
    COMMAND ${NVCC} -O3 -c -o ${RIFE_IO_OBJ}
        -I ${NVCC_INCLUDE_DIRECTORY}
        -gencode arch=compute_75,code=sm_75
        -gencode arch=compute_86,code=sm_86
        -gencode arch=compute_89,code=sm_89
        -gencode arch=compute_90,code=sm_90
        -gencode arch=compute_120,code=sm_120
        -gencode arch=compute_120,code=compute_120
        -Xcompiler "/MT /EHsc"
        ${CMAKE_CURRENT_SOURCE_DIR}/src/rife_io.cu
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/rife_io.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/rife_io.h
    VERBATIM
)
set_source_files_properties(${RIFE_IO_OBJ} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)

add_library(mlrt_trt SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mlrt_trt.cpp
    ${RIFE_IO_OBJ}
)

target_include_directories(mlrt_trt PRIVATE
    ${AVISYNTH_INCLUDE_DIRECTORY}
    ${CUDA_INCLUDE_DIRECTORY}
    ${CUDA_INCLUDE_DIRECTORY}/targets/x64
    ${CUDA_INCLUDE_DIRECTORY}/targets/x64/cccl
    ${TENSORRT_HOME}/include
)

set(CUDA_LIBRARY_DIRECTORY "" CACHE PATH "Path to cudart.lib")

target_link_directories(mlrt_trt PRIVATE ${TENSORRT_HOME}/lib ${CUDA_LIBRARY_DIRECTORY})

target_link_libraries(mlrt_trt PRIVATE
    ${AVISYNTH_LIBRARY}
    nvinfer_10
    cudart
)

target_compile_features(mlrt_trt PRIVATE cxx_std_20)

if (MSVC)
    target_compile_options(mlrt_trt PRIVATE /EHsc)
    target_link_libraries(mlrt_trt PRIVATE delayimp)
    target_link_options(mlrt_trt PRIVATE "/DELAYLOAD:nvinfer_10.dll" "/DELAYLOAD:AviSynth.dll")
    # Statically link the VC++ CRT so mlrt_trt.dll has no runtime dependency on
    # msvcp140/vcruntime140/vcruntime140_1.dll -- those are the redistributable, NOT part of
    # Windows itself, and a target deployment machine (e.g. a render-farm node) may not have
    # the VC++ redistributable installed. cudart64_*.dll/nvinfer_10.dll can't be statically
    # linked (vendor-shipped shared libs only), so those still ship as DLLs -- see the
    # explicit LoadLibraryExA preloading in mlrt_trt.cpp for how those are made portable instead.
    set_property(TARGET mlrt_trt PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if (WIN32)
    set_target_properties(mlrt_trt PROPERTIES
        PREFIX ""
        OUTPUT_NAME "mlrt_trt"
    )
endif()

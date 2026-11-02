#ifndef VSTRT_TRT_UTILS_H_
#define VSTRT_TRT_UTILS_H_

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <iostream>
#include <optional>
#include <string>
#include <variant>

#include <cuda_runtime.h>
#include <NvInferRuntime.h>

#include "cuda_helper.h"
#include "cuda_utils.h"
#include "rife_io.h"

using ErrorMessage = std::string;

struct RequestedTileSize {
    int tile_w;
    int tile_h;
};

struct VideoSize {
    int width;
    int height;
};

using TileSize = std::variant<RequestedTileSize, VideoSize>;

// Per-instance buffers of the YUV mode (raw planar YUV in/out, conversion to/from the engine's fp16 RGB on the GPU,
// see rife_io.h). Inactive (active == false) in the legacy fp32-RGB mode.
struct YuvResources {
    bool active{ false };
    // Zero-copy: the pack/unpack kernels read/write the pinned host buffers directly over UVA instead of using memcpy
    // nodes, so no transfer goes through the in-order copy-engine queue (which serialised copies behind other streams'
    // kernels).
    bool zc_in{ false };     // kernels read the pinned input buffers directly (no H2D memcpy)
    bool zc_out{ false };    // the unpack kernel writes the pinned output buffer directly (no D2H memcpy)
    // GPU timeline (MLRT_TRT_TIMELINE=1): events recorded between the phases of one inference
    //   0 start, 1 raw YUV uploaded, 2 packed, 3 TensorRT done, 4 unpacked, 5 downloaded
    bool timeline{ false };
    cudaEvent_t ev[6]{};
    uint8_t * dp_in[2]{};    // device-visible addresses of the mapped host buffers (zero-copy only)
    uint8_t * dp_out{};
    RifeIoFormat fmt{};
    size_t y_bytes{ 0 }, c_bytes{ 0 }, frame_bytes{ 0 };   // per frame, tightly packed planes Y, U, V
    Resource<uint8_t *, cudaFreeHost> h_in[2];             // pinned, written by the filter from the two input frames
    Resource<uint8_t *, cudaFree> d_in[2];
    Resource<uint8_t *, cudaFreeHost> h_out;               // pinned, read by the filter into the output frame
    Resource<uint8_t *, cudaFree> d_out;
    Resource<uint8_t *, cudaFreeHost> h_ts;                // one float: the timestep
    Resource<uint8_t *, cudaFree> d_ts;
};

struct InferenceInstance {
    MemoryResource src;
    MemoryResource dst;
    StreamResource stream;
    std::unique_ptr<nvinfer1::IExecutionContext> exec_context;
    GraphExecResource graphexec;
    YuvResources yuv;

#if NV_TENSORRT_MAJOR >= 10 || defined(TRT_MAJOR_RTX)
    Resource<uint8_t *, cudaFree> d_context_allocation;
#endif
};

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= verbosity) {
            std::cerr << message << '\n';
        }
    }

public:
    Logger() = default;

    void set_verbosity(Severity value) noexcept {
        this->verbosity = value;
    }

private:
    Severity verbosity;
};

static inline
std::optional<int> selectProfile(
    const std::unique_ptr<nvinfer1::ICudaEngine> & engine,
    const TileSize & tile_size,
    int batch_size = 1
) noexcept {

    int tile_w, tile_h;
    if (std::holds_alternative<RequestedTileSize>(tile_size)) {
        tile_w = std::get<RequestedTileSize>(tile_size).tile_w;
        tile_h = std::get<RequestedTileSize>(tile_size).tile_h;
    } else {
        tile_w = std::get<VideoSize>(tile_size).width;
        tile_h = std::get<VideoSize>(tile_size).height;
    }

#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    auto input_name = engine->getIOTensorName(0);
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)

    // finds the optimal profile
    for (int i = 0; i < engine->getNbOptimizationProfiles(); ++i) {
#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
        nvinfer1::Dims opt_dims = engine->getProfileShape(
            input_name, i, nvinfer1::OptProfileSelector::kOPT
        );
#else // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
        nvinfer1::Dims opt_dims = engine->getProfileDimensions(
            0, i, nvinfer1::OptProfileSelector::kOPT
        );
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)

        if (opt_dims.d[0] != batch_size) {
            continue;
        }
        if (opt_dims.d[2] == tile_h && opt_dims.d[3] == tile_w) {
            return i;
        }
    }

    // finds the first eligible profile
    for (int i = 0; i < engine->getNbOptimizationProfiles(); ++i) {
#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
        nvinfer1::Dims min_dims = engine->getProfileShape(
            input_name, i, nvinfer1::OptProfileSelector::kMIN
        );
#else // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
        nvinfer1::Dims min_dims = engine->getProfileDimensions(
            0, i, nvinfer1::OptProfileSelector::kMIN
        );
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)

        if (min_dims.d[0] > batch_size) {
            continue;
        }
        if (min_dims.d[2] > tile_h || min_dims.d[3] > tile_w) {
            continue;
        }

#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
        nvinfer1::Dims max_dims = engine->getProfileShape(
            input_name, i, nvinfer1::OptProfileSelector::kMAX
        );
#else // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
        nvinfer1::Dims max_dims = engine->getProfileDimensions(
            0, i, nvinfer1::OptProfileSelector::kMAX
        );
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)

        if (max_dims.d[0] < batch_size) {
            continue;
        }
        if (max_dims.d[2] < tile_h || max_dims.d[3] < tile_w) {
            continue;
        }

        return i;
    }

    // returns not-found
    return {};
}

static inline
std::optional<ErrorMessage> enqueue(
    const MemoryResource & src,
    const MemoryResource & dst,
    const std::unique_ptr<nvinfer1::IExecutionContext> & exec_context,
    cudaStream_t stream,
    const YuvResources * yuv = nullptr
) noexcept {

    const auto set_error = [](const ErrorMessage & message) {
        return message;
    };

    const bool is_yuv = yuv != nullptr && yuv->active;

    if (!is_yuv) {
        checkError(cudaMemcpyAsync(
            src.d_data, src.h_data, src.size,
            cudaMemcpyHostToDevice, stream
        ));
    } else {
        if (yuv->timeline) { cudaEventRecord(yuv->ev[0], stream); }
        if (!yuv->zc_in) {
            for (int i = 0; i < 2; ++i) {
                checkError(cudaMemcpyAsync(
                    yuv->d_in[i].data, yuv->h_in[i].data, yuv->frame_bytes,
                    cudaMemcpyHostToDevice, stream
                ));
            }
        }
        checkError(cudaMemcpyAsync(
            yuv->d_ts.data, yuv->h_ts.data, sizeof(float),
            cudaMemcpyHostToDevice, stream
        ));
        if (yuv->timeline) { cudaEventRecord(yuv->ev[1], stream); }
        checkError(rife_pack(
            yuv->fmt,
            rife_tight_in(yuv->fmt, yuv->zc_in ? yuv->dp_in[0] : yuv->d_in[0].data),
            rife_tight_in(yuv->fmt, yuv->zc_in ? yuv->dp_in[1] : yuv->d_in[1].data),
            reinterpret_cast<const float *>(yuv->d_ts.data),
            reinterpret_cast<__half *>(src.d_data.data), stream
        ));
        if (yuv->timeline) { cudaEventRecord(yuv->ev[2], stream); }
    }

#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    auto input_name = exec_context->getEngine().getIOTensorName(0);
    auto output_name = exec_context->getEngine().getIOTensorName(1);

    if (!exec_context->setTensorAddress(input_name, src.d_data.data)) {
        return set_error("set input tensor address failed");
    }
    if (!exec_context->setTensorAddress(output_name, dst.d_data.data)) {
        return set_error("set output tensor address failed");
    }
    if (!exec_context->enqueueV3(stream)) {
        return set_error("enqueue error");
    }
#else // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    void * bindings[] {
        static_cast<void *>(src.d_data.data),
        static_cast<void *>(dst.d_data.data)
    };

    if (!exec_context->enqueueV2(bindings, stream, nullptr)) {
        return set_error("enqueue error");
    }
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)

    if (!is_yuv) {
        checkError(cudaMemcpyAsync(
            dst.h_data, dst.d_data, dst.size,
            cudaMemcpyDeviceToHost, stream
        ));
    } else {
        if (yuv->timeline) { cudaEventRecord(yuv->ev[3], stream); }
        checkError(rife_unpack(
            yuv->fmt, reinterpret_cast<const __half *>(dst.d_data.data),
            rife_tight_out(yuv->fmt, yuv->zc_out ? yuv->dp_out : yuv->d_out.data), stream
        ));
        if (yuv->timeline) { cudaEventRecord(yuv->ev[4], stream); }
        if (!yuv->zc_out) {
            checkError(cudaMemcpyAsync(
                yuv->h_out.data, yuv->d_out.data, yuv->frame_bytes,
                cudaMemcpyDeviceToHost, stream
            ));
        }
        if (yuv->timeline) { cudaEventRecord(yuv->ev[5], stream); }
    }

    return {};
}

static inline
std::variant<ErrorMessage, GraphExecResource> getGraphExec(
    const MemoryResource & src, const MemoryResource & dst,
    const std::unique_ptr<nvinfer1::IExecutionContext> & exec_context,
    cudaStream_t stream,
    const YuvResources * yuv = nullptr
) noexcept {

    const auto set_error = [](const ErrorMessage & message) {
        return message;
    };

    // flush deferred internal state update
    // https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-821/developer-guide/index.html#cuda-graphs
    {
        auto result = enqueue(src, dst, exec_context, stream, yuv);
        if (result.has_value()) {
            return set_error(result.value());
        }
        checkError(cudaStreamSynchronize(stream));
    }

    checkError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    {
        auto result = enqueue(src, dst, exec_context, stream, yuv);
        if (result.has_value()) {
            return set_error(result.value());
        }
    }
    cudaGraph_t graph;
    checkError(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t graphexec;
    checkError(cudaGraphInstantiate(&graphexec, graph, nullptr, nullptr, 0));
    checkError(cudaGraphDestroy(graph));

    return graphexec;
}

static inline
size_t getSize(
    const nvinfer1::Dims & dim
) noexcept {

    size_t ret = 1;
    for (int i = 0; i < dim.nbDims; ++i) {
        ret *= dim.d[i];
    }
    return ret;
}

static inline
int getBytesPerSample(nvinfer1::DataType type) noexcept {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kBOOL:
            return 1;
        case nvinfer1::DataType::kUINT8:
            return 1;
#if (NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR) * 10 + NV_TENSORRT_PATCH >= 8061 || defined(TRT_MAJOR_RTX)
        case nvinfer1::DataType::kFP8:
            return 1;
#endif // (NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR) * 10 + NV_TENSORRT_PATCH >= 8061 || defined(TRT_MAJOR_RTX)
#if NV_TENSORRT_MAJOR >= 9 || defined(TRT_MAJOR_RTX)
        case nvinfer1::DataType::kBF16:
            return 2;
        case nvinfer1::DataType::kINT64:
            return 8;
#endif // NV_TENSORRT_MAJOR >= 9 || defined(TRT_MAJOR_RTX)
        default:
            return 0;
    }
}

static inline
std::variant<ErrorMessage, InferenceInstance> getInstance(
    const std::unique_ptr<nvinfer1::ICudaEngine> & engine,
    const std::optional<int> & profile_index,
    const TileSize & tile_size,
    bool use_cuda_graph,
#if NV_TENSORRT_MAJOR < 10 && !defined(TRT_MAJOR_RTX)
    bool & is_dynamic,
#else // NV_TENSORRT_MAJOR < 10 && !defined(TRT_MAJOR_RTX)
    bool is_dynamic,
#endif // NV_TENSORRT_MAJOR < 10 && !defined(TRT_MAJOR_RTX)
    const RifeIoFormat * yuv_fmt = nullptr
) noexcept {

    const auto set_error = [](const ErrorMessage & error_message) {
        return error_message;
    };

    StreamResource stream {};
    checkError(cudaStreamCreateWithFlags(&stream.data, cudaStreamNonBlocking));

    auto exec_context = std::unique_ptr<nvinfer1::IExecutionContext>(
#if NV_TENSORRT_MAJOR >= 10 || defined(TRT_MAJOR_RTX)
        engine->createExecutionContext(
            is_dynamic ?
            nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED :
            nvinfer1::ExecutionContextAllocationStrategy::kON_PROFILE_CHANGE
        )
#else
        engine->createExecutionContext()
#endif
    );

#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    auto input_name = engine->getIOTensorName(0);
    auto output_name = engine->getIOTensorName(1);
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)

    if (!exec_context->allInputDimensionsSpecified()) {
        if (!profile_index.has_value()) {
            return set_error("no valid optimization profile found");
        }
#if NV_TENSORRT_MAJOR < 10 && !defined(TRT_MAJOR_RTX)
        is_dynamic = true;
#endif // NV_TENSORRT_MAJOR < 10 && !defined(TRT_MAJOR_RTX)
        exec_context->setOptimizationProfileAsync(profile_index.value(), stream);
        checkError(cudaStreamSynchronize(stream));

#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
        nvinfer1::Dims dims = exec_context->getTensorShape(input_name);
#else // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
        nvinfer1::Dims dims = exec_context->getBindingDimensions(0);
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)

        dims.d[0] = 1;

        if (std::holds_alternative<RequestedTileSize>(tile_size)) {
            dims.d[2] = std::get<RequestedTileSize>(tile_size).tile_h;
            dims.d[3] = std::get<RequestedTileSize>(tile_size).tile_w;
        } else {
            dims.d[2] = std::get<VideoSize>(tile_size).height;
            dims.d[3] = std::get<VideoSize>(tile_size).width;
        }
#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
        exec_context->setInputShape(input_name, dims);
#else // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
        exec_context->setBindingDimensions(0, dims);
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    } else if (std::holds_alternative<RequestedTileSize>(tile_size)) {
#if NV_TENSORRT_MAJOR < 10 && !defined(TRT_MAJOR_RTX)
        is_dynamic = false;
#endif // NV_TENSORRT_MAJOR < 10 && !defined(TRT_MAJOR_RTX)

#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
        nvinfer1::Dims dims = exec_context->getTensorShape(input_name);
#else // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
        nvinfer1::Dims dims = exec_context->getBindingDimensions(0);
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)

        if (std::holds_alternative<RequestedTileSize>(tile_size)) {
            if (dims.d[2] != std::get<RequestedTileSize>(tile_size).tile_h ||
                dims.d[3] != std::get<RequestedTileSize>(tile_size).tile_w
            ) {
                return set_error("requested tile size not applicable");
            }
        } else {
            if (dims.d[2] != std::get<VideoSize>(tile_size).height ||
                dims.d[3] != std::get<VideoSize>(tile_size).width
            ) {
                return set_error("not supported video dimensions");
            }
        }
    }

    MemoryResource src {};
    {
#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
        auto dim = exec_context->getTensorShape(input_name);
        auto type = engine->getTensorDataType(input_name);
#else // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
        auto dim = exec_context->getBindingDimensions(0);
        auto type = engine->getBindingDataType(0);
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)

        auto size = getSize(dim) * getBytesPerSample(type);

        Resource<uint8_t *, cudaFree> d_data {};
        checkError(cudaMalloc(&d_data.data, size));

        Resource<uint8_t *, cudaFreeHost> h_data {};
        if (yuv_fmt == nullptr) {   // the YUV mode keeps its (much smaller) pinned buffers in YuvResources
            checkError(cudaMallocHost(&h_data.data, size, cudaHostAllocWriteCombined));
        }

        src = MemoryResource{
            .h_data = std::move(h_data),
            .d_data = std::move(d_data),
            .size=size
        };
    }

    MemoryResource dst {};
    {
#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
        auto dim = exec_context->getTensorShape(output_name);
        auto type = engine->getTensorDataType(output_name);
#else // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
        auto dim = exec_context->getBindingDimensions(1);
        auto type = engine->getBindingDataType(1);
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)

        auto size = getSize(dim) * getBytesPerSample(type);

        Resource<uint8_t *, cudaFree> d_data {};
        checkError(cudaMalloc(&d_data.data, size));

        Resource<uint8_t *, cudaFreeHost> h_data {};
        if (yuv_fmt == nullptr) {
            checkError(cudaMallocHost(&h_data.data, size));
        }

        dst = MemoryResource{
            .h_data = std::move(h_data),
            .d_data = std::move(d_data),
            .size=size
        };
    }

#if NV_TENSORRT_MAJOR >= 10 || defined(TRT_MAJOR_RTX)
    Resource<uint8_t *, cudaFree> d_context_allocation {};

    if (is_dynamic) {
        size_t buffer_size { exec_context->updateDeviceMemorySizeForShapes() };
        if (buffer_size == 0) {
            return set_error("failed to get internal activation buffer size");
        }

        checkError(cudaMalloc(&d_context_allocation.data, buffer_size));

#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 1001
        exec_context->setDeviceMemoryV2(d_context_allocation.data, static_cast<int64_t>(buffer_size));
#else // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 1001
        exec_context->setDeviceMemory(d_context_allocation.data);
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 1001
    }
#endif // NV_TENSORRT_MAJOR >= 10

    YuvResources yuv {};
    if (yuv_fmt != nullptr) {
        yuv.fmt = *yuv_fmt;
        const size_t bps = rife_bps(*yuv_fmt);
        yuv.y_bytes = static_cast<size_t>(yuv_fmt->width) * yuv_fmt->height * bps;
        yuv.c_bytes = static_cast<size_t>(rife_chroma_w(*yuv_fmt)) * rife_chroma_h(*yuv_fmt) * bps;
        yuv.frame_bytes = yuv.y_bytes + 2 * yuv.c_bytes;
        {
            const char * zc = std::getenv("MLRT_TRT_ZEROCOPY");   // "i", "o" or "io" (experiment; default off, see notes)
            const std::string z = zc ? zc : "";
            yuv.zc_in = z.find('i') != std::string::npos;
            yuv.zc_out = z.find('o') != std::string::npos;
        }
        const unsigned in_flags = cudaHostAllocWriteCombined | (yuv.zc_in ? cudaHostAllocMapped : 0u);
        const unsigned out_flags = yuv.zc_out ? cudaHostAllocMapped : 0u;
        for (int i = 0; i < 2; ++i) {
            checkError(cudaHostAlloc(reinterpret_cast<void **>(&yuv.h_in[i].data), yuv.frame_bytes, in_flags));
            checkError(cudaMalloc(&yuv.d_in[i].data, yuv.frame_bytes));
            if (yuv.zc_in) {
                checkError(cudaHostGetDevicePointer(reinterpret_cast<void **>(&yuv.dp_in[i]), yuv.h_in[i].data, 0));
            }
        }
        checkError(cudaHostAlloc(reinterpret_cast<void **>(&yuv.h_out.data), yuv.frame_bytes, out_flags));
        checkError(cudaMalloc(&yuv.d_out.data, yuv.frame_bytes));
        if (yuv.zc_out) {
            checkError(cudaHostGetDevicePointer(reinterpret_cast<void **>(&yuv.dp_out), yuv.h_out.data, 0));
        }
        checkError(cudaMallocHost(&yuv.h_ts.data, sizeof(float)));
        checkError(cudaMalloc(&yuv.d_ts.data, sizeof(float)));
        yuv.timeline = std::getenv("MLRT_TRT_TIMELINE") != nullptr;
        if (yuv.timeline) {
            for (auto & e : yuv.ev) {
                checkError(cudaEventCreate(&e));
            }
        }
        yuv.active = true;
    }

    GraphExecResource graphexec {};
    if (use_cuda_graph) {
        auto result = getGraphExec(
            src, dst,
            exec_context, stream,
            &yuv
        );
        if (std::holds_alternative<GraphExecResource>(result)) {
            graphexec = std::move(std::get<GraphExecResource>(result));
        } else {
            return set_error(std::get<ErrorMessage>(result));
        }
    }

    return InferenceInstance{
        .src = std::move(src),
        .dst = std::move(dst),
        .stream = std::move(stream),
        .exec_context = std::move(exec_context),
        .graphexec = std::move(graphexec),
        .yuv = std::move(yuv),
#if NV_TENSORRT_MAJOR >= 10 || defined(TRT_MAJOR_RTX)
        .d_context_allocation = std::move(d_context_allocation)
#endif
    };
}

static inline
std::optional<ErrorMessage> checkEngine(
    const std::unique_ptr<nvinfer1::ICudaEngine> & engine,
    bool flexible_output
) noexcept {

#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    int num_bindings = engine->getNbIOTensors();
#else // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    int num_bindings = engine->getNbBindings();
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)

    if (num_bindings != 2) {
        return "network binding count must be 2, got " + std::to_string(num_bindings);
    }

#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    auto input_name = engine->getIOTensorName(0);
    auto output_name = engine->getIOTensorName(1);
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)

#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    if (engine->getTensorIOMode(input_name) != nvinfer1::TensorIOMode::kINPUT) {
        return "the first binding should be an input binding";
    }
#else // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    if (!engine->bindingIsInput(0)) {
        return "the first binding should be an input binding";
    }
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)

#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    const nvinfer1::Dims & input_dims = engine->getTensorShape(input_name);
#else // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    const nvinfer1::Dims & input_dims = engine->getBindingDimensions(0);
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)

    if (input_dims.nbDims != 4) {
        return "expects network with 4-D input";
    }
    if (input_dims.d[0] != 1) {
        return "batch size of network input must be 1";
    }

#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    if (engine->getTensorIOMode(output_name) != nvinfer1::TensorIOMode::kOUTPUT) {
        return "the second binding should be an output binding";
    }
#else // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    if (engine->bindingIsInput(1)) {
        return "the second binding should be an output binding";
    }
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)

#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    const nvinfer1::Dims & output_dims = engine->getTensorShape(output_name);
#else // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    const nvinfer1::Dims & output_dims = engine->getBindingDimensions(1);
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)

    if (output_dims.nbDims != 4) {
        return "expects network with 4-D output";
    }
    if (output_dims.d[0] != 1) {
        return "batch size of network output must be 1";
    }

    auto out_channels = output_dims.d[1];
    if (out_channels != 1 && out_channels != 3 && !flexible_output) {
        return "output dimensions must be 1 or 3, or enable \"flexible_output\"";
    }

    auto in_height = input_dims.d[2];
    auto in_width = input_dims.d[3];
    auto out_height = output_dims.d[2];
    auto out_width = output_dims.d[3];
    if (out_height % in_height != 0 || out_width % in_width != 0) {
        return "output dimensions must be divisible by input dimensions";
    }

#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    for (const auto & name : { input_name, output_name }) {
        if (engine->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE) {
            return "network binding " + std::string{ name } + " should reside on device";
        }
#else // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    for (int i = 0; i < 2; i++) {
        if (engine->getLocation(i) != nvinfer1::TensorLocation::kDEVICE) {
            return "network binding " + std::to_string(i) + " should reside on device";
        }
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)

#if NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
        if (engine->getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR) {
            return "expects network IO with layout NCHW (row major linear)";
        }
#else // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
        if (engine->getBindingFormat(i) != nvinfer1::TensorFormat::kLINEAR) {
            return "expects network IO with layout NCHW (row major linear)";
        }
#endif // NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR >= 805 || defined(TRT_MAJOR_RTX)
    }

    return {};
}

static inline
std::variant<ErrorMessage, std::unique_ptr<nvinfer1::ICudaEngine>> initEngine(
    const char * engine_data, size_t engine_nbytes,
    const std::unique_ptr<nvinfer1::IRuntime> & runtime,
    bool flexible_output
) noexcept {

    const auto set_error = [](const ErrorMessage & error_message) {
        return error_message;
    };

    std::unique_ptr<nvinfer1::ICudaEngine> engine {
        runtime->deserializeCudaEngine(engine_data, engine_nbytes)
    };

    if (!engine) {
        return set_error("engine deserialization failed");
    }

    if (auto err = checkEngine(engine, flexible_output); err.has_value()) {
        return set_error(err.value());
    }

    return engine;
}

// 0: integer, 1: float
static inline
int getSampleType(nvinfer1::DataType type) noexcept {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
        case nvinfer1::DataType::kHALF:
#if (NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR) * 10 + NV_TENSORRT_PATCH >= 8061 || defined(TRT_MAJOR_RTX)
        case nvinfer1::DataType::kFP8:
#endif // (NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR) * 10 + NV_TENSORRT_PATCH >= 8061 || defined(TRT_MAJOR_RTX)
#if NV_TENSORRT_MAJOR >= 9 || defined(TRT_MAJOR_RTX)
        case nvinfer1::DataType::kBF16:
#endif // NV_TENSORRT_MAJOR >= 9 || defined(TRT_MAJOR_RTX)
            return 1;
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kINT32:
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kUINT8:
#if NV_TENSORRT_MAJOR >= 9 || defined(TRT_MAJOR_RTX)
        case nvinfer1::DataType::kINT64:
#endif // NV_TENSORRT_MAJOR >= 9 || defined(TRT_MAJOR_RTX)
            return 0;
        default:
            return -1;
    }
}

#endif // VSTRT_TRT_UTILS_H_

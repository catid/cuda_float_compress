#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "cuszplus_f32.h"

#include <vector>
#include <stdexcept>

torch::Tensor cuszplus_compress(torch::Tensor input, float max_error) {
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    if (input.dtype() != torch::kFloat32) {
        throw std::runtime_error("Input tensor must be of type float32");
    }

    const int float_count = static_cast<int>(input.numel());
    int compressed_buffer_size = GetCompressedBufferSize(float_count);

    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    torch::Tensor compresed_buffer = torch::empty(compressed_buffer_size, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    bool success = CompressFloats(
        stream,
        input.data_ptr<float>(),
        input.is_cuda(),
        float_count,
        compresed_buffer.data_ptr<uint8_t>(),
        compressed_buffer_size,
        2.f * max_error);

    if (!success) {
        throw std::runtime_error("compressor.Compress failed: float_count=" + std::to_string(float_count));
    }

    return compresed_buffer.slice(0, 0, compressed_buffer_size);
}

torch::Tensor cuszplus_decompress(torch::Tensor compressed, torch::Device device) {
    if (!compressed.is_contiguous()) {
        compressed = compressed.contiguous();
    }
    if (compressed.dtype() != torch::kUInt8) {
        throw std::runtime_error("Compressed tensor must be of type uint8");
    }
    if (compressed.is_cuda()) {
        throw std::runtime_error("Input tensor must be on the CPU");
    }
    if (!device.is_cuda()) {
        throw std::runtime_error("Output device must be CUDA");
    }

    const int compressed_bytes = static_cast<int>(compressed.numel());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int float_count = GetDecompressedFloatCount(
        compressed.data_ptr<unsigned char>(),
        compressed_bytes);
    if (float_count < 0) {
        float_count = 0;
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor result = torch::empty(float_count, options);

    if (float_count <= 0) {
        return result;
    }

    bool success = DecompressFloats(
        stream,
        compressed.data_ptr<unsigned char>(),
        compressed_bytes,
        result.data_ptr<float>());

    if (!success) {
        throw std::runtime_error("DecompressFloats failed");
    }

    return result;
}

PYBIND11_MODULE(cuda_float_compress, m) {
    m.def("cuszplus_compress", &cuszplus_compress, "cuszplus_compress");
    m.def("cuszplus_decompress", &cuszplus_decompress, "cuszplus_decompress");
}

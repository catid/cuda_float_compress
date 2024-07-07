#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "cuszplus_f32.h"

#include <vector>
#include <stdexcept>

torch::Tensor cuszplus_compress(torch::Tensor input, float errorBound) {
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    if (input.dtype() != torch::kFloat32) {
        throw std::runtime_error("Input tensor must be of type float32");
    }

    float* data = input.data_ptr<float>();
    size_t nbEle = input.numel();

    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(input.device());
    torch::Tensor cmpBytes = torch::empty(nbEle * sizeof(float), options);
    size_t cmpSize = 0;

    FloatCompressor compressor;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (!compressor.Compress(stream, data, input.is_cuda(), nbEle, errorBound)) {
        throw std::runtime_error("compressor.Compress failed: nbEle=" + std::to_string(nbEle));
    }

    return cmpBytes.slice(0, cmpSize);
}

torch::Tensor cuszplus_decompress(torch::Tensor compressed) {
    if (!compressed.is_contiguous()) {
        compressed = compressed.contiguous();
    }
    
    if (compressed.dtype() != torch::kUInt8) {
        throw std::runtime_error("Compressed tensor must be of type uint8");
    }

    if (compressed.is_cuda()) {
        throw std::runtime_error("Input tensor must be on the CPU");
    }

    const size_t cmpSize = compressed.numel();

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(compressed.device());

    FloatDecompressor decompressor;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (!decompressor.Decompress(stream, compressed.data_ptr<unsigned char>(), cmpSize)) {
        throw std::runtime_error("decompressor.Decompress failed");
    }

    // FIXME: Copy result back or allocate internally.

    torch::Tensor result = torch::empty(1, options);
    return result;
}

PYBIND11_MODULE(cuda_float_compress, m) {
    m.def("cuszplus_compress", &cuszplus_compress, "cuszplus_compress");
    m.def("cuszplus_decompress", &cuszplus_decompress, "cuszplus_decompress");
}

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cuSZp_utility.h>
#include <cuSZp_entry_f32.h>

#include <vector>
#include <stdexcept>

static void checkCudaErrors(cudaError_t err, const char* location) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(location) + ": " + cudaGetErrorString(err));
    }
}

torch::Tensor cuszp_compress(torch::Tensor input, float errorBound) {
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

    if (!input.is_cuda()) {
        // CPU processing
        SZp_compress_hostptr_f32(data, cmpBytes.data_ptr<unsigned char>(), nbEle, &cmpSize, errorBound);
    } else {
        // GPU processing
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        SZp_compress_deviceptr_f32(data, cmpBytes.data_ptr<unsigned char>(), nbEle, &cmpSize, errorBound, stream);
        checkCudaErrors(cudaGetLastError(), "SZp_compress_deviceptr_f32");
        checkCudaErrors(cudaStreamSynchronize(stream), "cudaStreamSynchronize after compression");
    }

    return cmpBytes.slice(0, cmpSize);
}

torch::Tensor cuszp_decompress(torch::Tensor compressed, size_t nbEle, size_t cmpSize, float errorBound) {
    if (!compressed.is_contiguous()) {
        compressed = compressed.contiguous();
    }
    
    if (compressed.dtype() != torch::kUInt8) {
        throw std::runtime_error("Compressed tensor must be of type uint8");
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(compressed.device());
    torch::Tensor result = torch::empty(nbEle, options);

    if (!compressed.is_cuda()) {
        // CPU processing
        SZp_decompress_hostptr_f32(result.data_ptr<float>(), compressed.data_ptr<unsigned char>(), nbEle, cmpSize, errorBound);
    } else {
        // GPU processing
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        SZp_decompress_deviceptr_f32(result.data_ptr<float>(), compressed.data_ptr<unsigned char>(), nbEle, cmpSize, errorBound, stream);
        checkCudaErrors(cudaGetLastError(), "SZp_compress_deviceptr_f32");
        checkCudaErrors(cudaStreamSynchronize(stream), "cudaStreamSynchronize after compression");
    }

    return result;
}

PYBIND11_MODULE(cuda_float_compress, m) {
    m.def("cuszp_compress", &cuszp_compress, "cuszp_compress");
    m.def("cuszp_decompress", &cuszp_decompress, "cuszp_decompress");
}

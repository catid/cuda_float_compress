#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <chrono>

#define BLOCK_SIZE 256

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

__global__ void combinedKernel(uint32_t* data) {
    using BlockScan = cub::BlockScan<uint32_t, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    data[idx] = 1;  // Initialize directly to 1

    __syncthreads(); // Barrier for smem reuse

    // Inclusive Sum (using CUB)
    uint32_t thread_data = data[idx]; // Load the initialized value
    BlockScan block_scan(temp_storage);
    block_scan.InclusiveSum(thread_data, thread_data);

    data[idx] = thread_data;

    __syncthreads(); // Barrier for smem reuse
}

static int RoundUpBlocks(int x) { 
    return (x + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
}

int main() {
    const int NUM_ITEMS = 1000000;
    int numBlocks = (NUM_ITEMS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int globalMemoryBytes = RoundUpBlocks(NUM_ITEMS) * sizeof(uint32_t);

    // Allocate device array
    uint32_t* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, globalMemoryBytes));

    // Start timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    auto cpuStart = std::chrono::high_resolution_clock::now();

    // Initialize array with 1s using a kernel
    combinedKernel<<<numBlocks, BLOCK_SIZE>>>(d_data);

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuDuration = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart);

    // Calculate and print the elapsed times
    float gpuMilliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpuMilliseconds, start, stop));
    std::cout << "GPU kernel execution time: " << gpuMilliseconds << " ms (" 
              << gpuMilliseconds * 1000 << " us)" << std::endl;
    std::cout << "CPU (including kernel launch & overhead) time: " << cpuDuration.count() << " us" << std::endl;

    // Copy result back to host for verification
    std::vector<uint32_t> h_result(NUM_ITEMS);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_data, NUM_ITEMS * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Verify the result on CPU
    bool correct = true;
    uint32_t expected_sum = 0;
    for (int i = 0; i < NUM_ITEMS; ++i) {
        expected_sum += 1;
        if (h_result[i] != expected_sum) {
            std::cout << "Error at index " << i << ": expected " << expected_sum << ", got " << h_result[i] << std::endl;
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "Prefix sum is correct!" << std::endl;
        std::cout << "Last element (sum of all elements): " << h_result[NUM_ITEMS - 1] << std::endl;
    } else {
        std::cout << "Prefix sum is incorrect." << std::endl;
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}

#include <cuda_runtime.h>

#include <iostream>
#include <cassert>
#include <cstdint>
#include <vector>
#include <functional>
#include <chrono>
#include <algorithm>
#include <iomanip>
using namespace std;


//------------------------------------------------------------------------------
// Constants

const int NUM_RUNS = 20;  // Number of times to run each test


//------------------------------------------------------------------------------
// Tools

// Error checking macro
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

struct CallbackScope {
    CallbackScope(std::function<void()> func) : func(func) {}
    ~CallbackScope() { func(); }
    std::function<void()> func;
};

double median(std::vector<double>& v) {
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}


//------------------------------------------------------------------------------
// GPU Kernels

__device__ void interleave_words_1bit(
    const uint32_t* const __restrict__ input,
    uint32_t* const __restrict__ output)
{
    #pragma unroll
    for (uint32_t shift = 0; shift < 32; shift++) {
        uint32_t mask = 1U << shift;
        uint32_t result = (input[0] & mask) >> shift;

        #pragma unroll
        for (uint32_t i = 1; i < 32; ++i) {
            result |= ((input[i] & mask) >> shift) << i;
        }

        output[shift] = result;
    }
}

__device__ void interleave_words_2bit(
    const uint32_t* const __restrict__ input,
    uint32_t* const __restrict__ output)
{
    #pragma unroll
    for (uint32_t shift = 0; shift < 32; shift += 2) {
        uint32_t result_0 = 0;
        uint32_t result_1 = 0;
        uint32_t mask = 0x3 << shift;

        #pragma unroll
        for (uint32_t i = 0; i < 16; ++i) {
            uint32_t bits_0 = (input[i] & mask) >> shift;
            uint32_t bits_1 = (input[i + 16] & mask) >> shift;
            result_0 |= (bits_0 << (i * 2));
            result_1 |= (bits_1 << (i * 2));
        }

        output[shift] = result_0;
        output[shift + 1] = result_1;
    }
}

__device__ void interleave_words_4bit(
    const uint32_t* const __restrict__ input,
    uint32_t* const __restrict__ output)
{
    #pragma unroll
    for (uint32_t shift = 0; shift < 32; shift += 4) {
        uint32_t result_0 = 0;
        uint32_t result_1 = 0;
        uint32_t result_2 = 0;
        uint32_t result_3 = 0;
        uint32_t mask = 0xF << shift;

        #pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            uint32_t bits_0 = (input[i] & mask) >> shift;
            uint32_t bits_1 = (input[i + 8] & mask) >> shift;
            uint32_t bits_2 = (input[i + 16] & mask) >> shift;
            uint32_t bits_3 = (input[i + 24] & mask) >> shift;
            
            result_0 |= (bits_0 << (i * 4));
            result_1 |= (bits_1 << (i * 4));
            result_2 |= (bits_2 << (i * 4));
            result_3 |= (bits_3 << (i * 4));
        }

        output[shift] = result_0;
        output[shift + 1] = result_1;
        output[shift + 2] = result_2;
        output[shift + 3] = result_3;
    }
}

__device__ void interleave_words_8bit(
    const uint32_t* const __restrict__ input,
    uint32_t* const __restrict__ output)
{
    #pragma unroll
    for (uint32_t shift = 0; shift < 32; shift += 8) {
        uint32_t mask = 0xFF << shift;

        uint32_t result_0 = 0, result_1 = 0, result_2 = 0, result_3 = 0;
        uint32_t result_4 = 0, result_5 = 0, result_6 = 0, result_7 = 0;

        #pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            result_0 |= (((input[i] & mask) >> shift) << (i * 8));
            result_1 |= (((input[i + 4] & mask) >> shift) << (i * 8));
            result_2 |= (((input[i + 8] & mask) >> shift) << (i * 8));
            result_3 |= (((input[i + 12] & mask) >> shift) << (i * 8));
            result_4 |= (((input[i + 16] & mask) >> shift) << (i * 8));
            result_5 |= (((input[i + 20] & mask) >> shift) << (i * 8));
            result_6 |= (((input[i + 24] & mask) >> shift) << (i * 8));
            result_7 |= (((input[i + 28] & mask) >> shift) << (i * 8));
        }

        output[shift] = result_0;
        output[shift + 1] = result_1;
        output[shift + 2] = result_2;
        output[shift + 3] = result_3;
        output[shift + 4] = result_4;
        output[shift + 5] = result_5;
        output[shift + 6] = result_6;
        output[shift + 7] = result_7;
    }
}


//------------------------------------------------------------------------------
// Interleave Kernel

__global__ void interleave_kernel_1bit(
    const uint32_t* input,
    uint32_t* output)
{
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    input += thread_idx * 32;
    output += thread_idx * 32;

    uint32_t words[32];

    #pragma unroll
    for (uint32_t j = 0; j < 32; j++) {
        words[j] = input[j];
    }

    interleave_words_1bit(words, output);
}


//------------------------------------------------------------------------------
// CPU Version

// Helper function to pack bits from 32 input words into a single output word
uint32_t cpu_pack_bits(const uint32_t* input, uint32_t bit_position) {
    uint32_t result = 0;
    uint32_t mask = 1u << bit_position;
    
    for (int i = 0; i < 32; ++i) {
        // Extract the bit at bit_position from each input word
        uint32_t bit = (input[i] & mask) ? 1 : 0;
        
        // Place this bit in the correct position in the result
        result |= (bit << i);
    }
    
    return result;
}

// Main interleave function
void cpu_interleave_1bit(const uint32_t* input, uint32_t* output, int block_count) {
    for (int block = 0; block < block_count; ++block) {
        // Pointer to the start of the current input block
        const uint32_t* block_input = input + (block * 32);
        
        // Pointer to the start of the current output block
        uint32_t* block_output = output + (block * 32);
        
        // For each bit position in the 32-bit words
        for (int bit_pos = 0; bit_pos < 32; ++bit_pos) {
            // Pack the bits at this position from all 32 input words in the current block
            block_output[bit_pos] = cpu_pack_bits(block_input, bit_pos);
        }
    }
}


//------------------------------------------------------------------------------
// GPU Interleave Kernel (2 bits at a time)

__global__ void interleave_kernel_2bit(
    const uint32_t* input,
    uint32_t* output)
{
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    input += thread_idx * 32;
    output += thread_idx * 32;

    // Copy input to local register array
    uint32_t words[32];
    #pragma unroll
    for (uint32_t i = 0; i < 32; ++i) {
        words[i] = input[i];
    }

    interleave_words_2bit(words, output);
}


//------------------------------------------------------------------------------
// CPU Version (2 bits at a time)

// Helper function to pack 2 bits from 16 input words into a single output word
uint32_t cpu_pack_2bits(const uint32_t* input, uint32_t shift) {
    uint32_t result = 0;
    uint32_t mask = 0x3 << shift;
    
    for (int i = 0; i < 16; ++i) {
        // Extract 2 bits at shift position from each input word
        uint32_t bits = (input[i] & mask) >> shift;
        
        // Place these 2 bits in the correct position in the result
        result |= (bits << (i * 2));
    }
    
    return result;
}

// Main interleave function
void cpu_interleave_2bit(const uint32_t* input, uint32_t* output, int block_count) {
    for (int block = 0; block < block_count; ++block) {
        // Pointer to the start of the current input block
        const uint32_t* block_input = input + (block * 32);
        
        // Pointer to the start of the current output block
        uint32_t* block_output = output + (block * 32);
        
        // For each 2-bit position in the 32-bit words
        for (int shift = 0; shift < 32; shift += 2) {
            // Pack 2 bits at this position from all 32 input words in the current block
            block_output[shift] = cpu_pack_2bits(block_input, shift);
            block_output[shift + 1] = cpu_pack_2bits(block_input + 16, shift);
        }
    }
}


//------------------------------------------------------------------------------
// GPU Interleave Kernel (4 bits at a time)

__global__ void interleave_kernel_4bit(
    const uint32_t* input,
    uint32_t* output)
{
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    input += thread_idx * 32;
    output += thread_idx * 32;

    // Copy input to local register array
    uint32_t words[32];
    #pragma unroll
    for (uint32_t i = 0; i < 32; ++i) {
        words[i] = input[i];
    }

    interleave_words_4bit(words, output);
}


//------------------------------------------------------------------------------
// CPU Version (4 bits at a time)

// Helper function to pack 4 bits from 8 input words into a single output word
uint32_t cpu_pack_4bits(const uint32_t* input, uint32_t shift) {
    uint32_t result = 0;
    uint32_t mask = 0xF << shift;
    
    for (int i = 0; i < 8; ++i) {
        // Extract 4 bits at shift position from each input word
        uint32_t bits = (input[i] & mask) >> shift;
        
        // Place these 4 bits in the correct position in the result
        result |= (bits << (i * 4));
    }
    
    return result;
}

// Main interleave function
void cpu_interleave_4bit(const uint32_t* input, uint32_t* output, int block_count) {
    for (int block = 0; block < block_count; ++block) {
        // Pointer to the start of the current input block
        const uint32_t* block_input = input + (block * 32);
        
        // Pointer to the start of the current output block
        uint32_t* block_output = output + (block * 32);
        
        // For each 4-bit position in the 32-bit words
        for (int shift = 0; shift < 32; shift += 4) {
            // Pack 4 bits at this position from all 32 input words in the current block
            block_output[shift] = cpu_pack_4bits(block_input, shift);
            block_output[shift + 1] = cpu_pack_4bits(block_input + 8, shift);
            block_output[shift + 2] = cpu_pack_4bits(block_input + 16, shift);
            block_output[shift + 3] = cpu_pack_4bits(block_input + 24, shift);
        }
    }
}


//------------------------------------------------------------------------------
// GPU Interleave Kernel (8 bits at a time)

__global__ void interleave_kernel_8bit(
    const uint32_t* input,
    uint32_t* output)
{
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    input += thread_idx * 32;
    output += thread_idx * 32;

    // Copy input to local register array
    uint32_t words[32];
    #pragma unroll
    for (uint32_t i = 0; i < 32; ++i) {
        words[i] = input[i];
    }

    interleave_words_8bit(words, output);
}


//------------------------------------------------------------------------------
// CPU Version (8 bits at a time)

// Helper function to pack 8 bits from 4 input words into a single output word
uint32_t cpu_pack_8bits(const uint32_t* input, uint32_t shift) {
    uint32_t result = 0;
    uint32_t mask = 0xFF << shift;
    
    for (int i = 0; i < 4; ++i) {
        // Extract 8 bits at shift position from each input word
        uint32_t bits = (input[i] & mask) >> shift;
        
        // Place these 8 bits in the correct position in the result
        result |= (bits << (i * 8));
    }
    
    return result;
}

// Main interleave function
void cpu_interleave_8bit(const uint32_t* input, uint32_t* output, int block_count) {
    for (int block = 0; block < block_count; ++block) {
        // Pointer to the start of the current input block
        const uint32_t* block_input = input + (block * 32);

        // Pointer to the start of the current output block
        uint32_t* block_output = output + (block * 32);

        // For each 8-bit position in the 32-bit words
        for (int shift = 0; shift < 32; shift += 8) {
            // Pack 8 bits at this position from all 32 input words in the current block
            block_output[shift] = cpu_pack_8bits(block_input, shift);
            block_output[shift + 1] = cpu_pack_8bits(block_input + 4, shift);
            block_output[shift + 2] = cpu_pack_8bits(block_input + 8, shift);
            block_output[shift + 3] = cpu_pack_8bits(block_input + 12, shift);
            block_output[shift + 4] = cpu_pack_8bits(block_input + 16, shift);
            block_output[shift + 5] = cpu_pack_8bits(block_input + 20, shift);
            block_output[shift + 6] = cpu_pack_8bits(block_input + 24, shift);
            block_output[shift + 7] = cpu_pack_8bits(block_input + 28, shift);
        }
    }
}


//------------------------------------------------------------------------------
// Tests

void run_test(int bits) {
    const int value_count = 2 * 1024 * 1024;
    const int interleaved_count = 32;
    const int operation_count = value_count / interleaved_count;
    const int block_size = 256;
    const int block_count = operation_count / block_size;

    // Allocate device memory
    uint32_t *d_input, *d_output;
    cudaMallocManaged(&d_input, value_count * sizeof(uint32_t));
    CallbackScope input_cleanup([&]() { cudaFree(d_input); });
    cudaMallocManaged(&d_output, value_count * sizeof(uint32_t));
    CallbackScope output_cleanup([&]() { cudaFree(d_output); });
    cudaCheckError();

    for (int i = 0; i < value_count; i++) {
        d_input[i] = rand();
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<double> gpu_times, cpu_times;

    int num_runs = NUM_RUNS;
    if (bits == 1) {
        num_runs = 3;
    }

    for (int run = 0; run < num_runs; run++) {
        // Run the kernel
        cudaEventRecord(start);
        if (bits == 1) {
            interleave_kernel_1bit<<<block_count, block_size>>>(d_input, d_output);
        } else if (bits == 2) {
            interleave_kernel_2bit<<<block_count, block_size>>>(d_input, d_output);
        } else if (bits == 4) {
            interleave_kernel_4bit<<<block_count, block_size>>>(d_input, d_output);
        } else if (bits == 8) {
            interleave_kernel_8bit<<<block_count, block_size>>>(d_input, d_output);
        }
        cudaEventRecord(stop);

        cudaDeviceSynchronize();
        cudaCheckError();

        float gpu_milliseconds = 0;
        cudaEventElapsedTime(&gpu_milliseconds, start, stop);
        gpu_times.push_back(gpu_milliseconds);

        // CPU Timing
        std::vector<uint32_t> expected_output(value_count);
        auto cpu_start = std::chrono::high_resolution_clock::now();
        if (bits == 1) {
            cpu_interleave_1bit(d_input, expected_output.data(), operation_count);
        } else if (bits == 2) {
            cpu_interleave_2bit(d_input, expected_output.data(), operation_count);
        } else if (bits == 4) {
            cpu_interleave_4bit(d_input, expected_output.data(), operation_count);
        } else if (bits == 8) {
            cpu_interleave_8bit(d_input, expected_output.data(), operation_count);
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
        cpu_times.push_back(cpu_duration.count() / 1000.0);

        // Verify results (only for the first run)
        if (run == 0) {
            for (int i = 0; i < value_count; i++) {
                if (d_output[i] != expected_output[i]) {
                    cerr << "Mismatch at index " << i << " (" << bits << " bits):\n";
                    cerr << "Expected: 0x" << std::hex << expected_output[i] << std::dec << endl;
                    cerr << "Got:      0x" << std::hex << d_output[i] << std::dec << endl;
                    assert(false);
                }
            }
        }
    }

    double median_gpu_time = median(gpu_times);
    double median_cpu_time = median(cpu_times);

    cout << "Test passed for " << bits << "-bit interleave with " << value_count << " values" << endl;
    cout << "Median GPU Time: " << std::fixed << std::setprecision(3) << median_gpu_time << " ms" << endl;
    cout << "Median CPU Time: " << std::fixed << std::setprecision(3) << median_cpu_time << " ms" << endl;
    cout << "GPU Speedup: " << std::fixed << std::setprecision(2) << (median_cpu_time / median_gpu_time) << "x" << endl;
    cout << endl;
}

int main() {
    cout << "Running tests " << NUM_RUNS << " times each and reporting median times:" << endl << endl;
    run_test(1);
    run_test(2);
    run_test(4);
    run_test(8);

    cout << "All tests passed!" << endl;
    return 0;
}

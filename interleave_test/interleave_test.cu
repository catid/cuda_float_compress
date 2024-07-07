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

static double median(std::vector<double>& v) {
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}

static uint32_t generate_mask(int bits) {
    if (bits == 0) return 0;
    if (bits >= 32) return 0xFFFFFFFF;
    return (1U << bits) - 1;
}


//------------------------------------------------------------------------------
// GPU Kernels

__device__ inline void interleave_words_1bit(
    const uint32_t* const __restrict__ input,
    uint32_t* const __restrict__ output,
    uint32_t bits)
{
    #pragma unroll
    for (uint32_t shift = 0; shift < bits; shift++) {
        uint32_t mask = 1U << shift;
        uint32_t result = (input[0] & mask) >> shift;

        #pragma unroll
        for (uint32_t i = 1; i < 32; ++i) {
            result |= ((input[i] & mask) >> shift) << i;
        }

        output[shift] = result;
    }
}

__device__ inline void interleave_words_2bit(
    const uint32_t* const __restrict__ input,
    uint32_t* const __restrict__ output,
    uint32_t bits)
{
    #pragma unroll
    for (uint32_t shift = 0; shift < bits; shift += 2) {
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

__device__ inline void interleave_words_4bit(
    const uint32_t* const __restrict__ input,
    uint32_t* const __restrict__ output,
    uint32_t bits)
{
    #pragma unroll
    for (uint32_t shift = 0; shift < bits; shift += 4) {
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

__device__ inline void interleave_words_8bit(
    const uint32_t* const __restrict__ input,
    uint32_t* const __restrict__ output,
    uint32_t bits)
{
    #pragma unroll
    for (uint32_t shift = 0; shift < bits; shift += 8) {
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
    uint32_t* output,
    uint32_t bits)
{
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    input += thread_idx * 32;
    output += thread_idx * bits;

    uint32_t words[32];

    #pragma unroll
    for (uint32_t j = 0; j < 32; j++) {
        words[j] = input[j];
    }

    interleave_words_1bit(words, output, bits);
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
void cpu_interleave_1bit(const uint32_t* input, uint32_t* output, int block_count, int bits) {
    for (int block = 0; block < block_count; ++block) {
        // Pointer to the start of the current input block
        const uint32_t* block_input = input + (block * 32);
        
        // Pointer to the start of the current output block
        uint32_t* block_output = output + (block * bits);
        
        // For each bit position in the 32-bit words
        for (int bit_pos = 0; bit_pos < bits; ++bit_pos) {
            // Pack the bits at this position from all 32 input words in the current block
            block_output[bit_pos] = cpu_pack_bits(block_input, bit_pos);
        }
    }
}


//------------------------------------------------------------------------------
// GPU Interleave Kernel (2 bits at a time)

__global__ void interleave_kernel_2bit(
    const uint32_t* input,
    uint32_t* output,
    uint32_t bits)
{
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    input += thread_idx * 32;
    output += thread_idx * bits;

    // Copy input to local register array
    uint32_t words[32];
    #pragma unroll
    for (uint32_t i = 0; i < 32; ++i) {
        words[i] = input[i];
    }

    interleave_words_2bit(words, output, bits);
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
void cpu_interleave_2bit(const uint32_t* input, uint32_t* output, int block_count, int bits) {
    for (int block = 0; block < block_count; ++block) {
        // Pointer to the start of the current input block
        const uint32_t* block_input = input + (block * 32);

        // Pointer to the start of the current output block
        uint32_t* block_output = output + (block * bits);

        // For each 2-bit position in the 32-bit words
        for (int shift = 0; shift < bits; shift += 2) {
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
    uint32_t* output,
    uint32_t bits)
{
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    input += thread_idx * 32;
    output += thread_idx * bits;

    // Copy input to local register array
    uint32_t words[32];
    #pragma unroll
    for (uint32_t i = 0; i < 32; ++i) {
        words[i] = input[i];
    }

    interleave_words_4bit(words, output, bits);
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
void cpu_interleave_4bit(const uint32_t* input, uint32_t* output, int block_count, int bits) {
    for (int block = 0; block < block_count; ++block) {
        // Pointer to the start of the current input block
        const uint32_t* block_input = input + (block * 32);
        
        // Pointer to the start of the current output block
        uint32_t* block_output = output + (block * bits);
        
        // For each 4-bit position in the 32-bit words
        for (int shift = 0; shift < bits; shift += 4) {
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
    uint32_t* output,
    uint32_t bits)
{
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    input += thread_idx * 32;
    output += thread_idx * bits;

    // Copy input to local register array
    uint32_t words[32];
    #pragma unroll
    for (uint32_t i = 0; i < 32; ++i) {
        words[i] = input[i];
    }

    interleave_words_8bit(words, output, bits);
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
void cpu_interleave_8bit(const uint32_t* input, uint32_t* output, int block_count, int bits) {
    for (int block = 0; block < block_count; ++block) {
        // Pointer to the start of the current input block
        const uint32_t* block_input = input + (block * 32);

        // Pointer to the start of the current output block
        uint32_t* block_output = output + (block * bits);

        // For each 8-bit position in the 32-bit words
        for (int shift = 0; shift < bits; shift += 8) {
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

void run_test(int bits, int low_bits) {
    const int value_count = 2 * 1024 * 1024;
    const int interleaved_count = 32;
    const int operation_count = value_count / interleaved_count;
    const int block_size = 256;
    const int block_count = operation_count / block_size;

    // Allocate device memory
    uint32_t *d_input, *d_output;
    cudaMallocManaged(&d_input, value_count * sizeof(uint32_t));
    CallbackScope input_cleanup([&]() { cudaFree(d_input); });
    cudaMallocManaged(&d_output, value_count * sizeof(uint32_t) * low_bits / 32);
    CallbackScope output_cleanup([&]() { cudaFree(d_output); });
    cudaCheckError();

    uint32_t mask = generate_mask(low_bits);
    for (int i = 0; i < value_count; i++) {
        d_input[i] = rand() & mask;
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
    if (low_bits != 32 && low_bits != 16) {
        num_runs = 1;
    }

    for (int run = 0; run < num_runs; run++) {
        // Run the kernel
        cudaEventRecord(start);
        if (bits == 1) {
            interleave_kernel_1bit<<<block_count, block_size>>>(d_input, d_output, low_bits);
        } else if (bits == 2) {
            interleave_kernel_2bit<<<block_count, block_size>>>(d_input, d_output, low_bits);
        } else if (bits == 4) {
            interleave_kernel_4bit<<<block_count, block_size>>>(d_input, d_output, low_bits);
        } else if (bits == 8) {
            interleave_kernel_8bit<<<block_count, block_size>>>(d_input, d_output, low_bits);
        }
        cudaEventRecord(stop);

        cudaDeviceSynchronize();
        cudaCheckError();

        float gpu_milliseconds = 0;
        cudaEventElapsedTime(&gpu_milliseconds, start, stop);
        gpu_times.push_back(gpu_milliseconds);

        // CPU Timing
        std::vector<uint32_t> expected_output(value_count * low_bits / 32);
        auto cpu_start = std::chrono::high_resolution_clock::now();
        if (bits == 1) {
            cpu_interleave_1bit(d_input, expected_output.data(), operation_count, low_bits);
        } else if (bits == 2) {
            cpu_interleave_2bit(d_input, expected_output.data(), operation_count, low_bits);
        } else if (bits == 4) {
            cpu_interleave_4bit(d_input, expected_output.data(), operation_count, low_bits);
        } else if (bits == 8) {
            cpu_interleave_8bit(d_input, expected_output.data(), operation_count, low_bits);
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
        cpu_times.push_back(cpu_duration.count() / 1000.0);

        // Verify results (only for the first run)
        if (run == 0) {
            for (int i = 0; i < value_count * low_bits / 32; i++) {
                if (d_output[i] != expected_output[i]) {
                    cerr << "Mismatch at index " << i << " (" << bits << " bits, low_bits=" << low_bits << "):\n";
                    cerr << "Expected: 0x" << std::hex << expected_output[i] << std::dec << endl;
                    cerr << "Got:      0x" << std::hex << d_output[i] << std::dec << endl;
                    assert(false);
                }
            }
        }
    }

    double median_gpu_time = median(gpu_times);
    double median_cpu_time = median(cpu_times);

    cout << "Test passed for " << bits << "-bit interleave with " << value_count << " values, low_bits=" << low_bits << endl;
    cout << "Median GPU Time: " << std::fixed << std::setprecision(3) << median_gpu_time << " ms" << endl;
    cout << "Median CPU Time: " << std::fixed << std::setprecision(3) << median_cpu_time << " ms" << endl;
    cout << "GPU Speedup: " << std::fixed << std::setprecision(2) << (median_cpu_time / median_gpu_time) << "x" << endl;
    cout << endl;
}


//------------------------------------------------------------------------------
// GPU De-interleave Kernels

__device__ inline void deinterleave_words_1bit(
    const uint32_t* const __restrict__ input,
    uint32_t* const __restrict__ output,
    uint32_t bits)
{
    #pragma unroll
    for (uint32_t i = 0; i < 32; i++) {
        uint32_t result = 0;
        #pragma unroll
        for (uint32_t j = 0; j < bits; j++) {
            result |= ((input[j] >> i) & 1) << j;
        }
        output[i] = result;
    }
}

__device__ inline void deinterleave_words_2bit(
    const uint32_t* const __restrict__ input,
    uint32_t* const __restrict__ output,
    uint32_t bits)
{
    #pragma unroll
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t result_0 = 0, result_1 = 0;
        #pragma unroll
        for (uint32_t j = 0; j < bits; j += 2) {
            result_0 |= ((input[j] >> (i*2)) & 3) << j;
            result_1 |= ((input[j+1] >> (i*2)) & 3) << j;
        }
        output[i] = result_0;
        output[i + 16] = result_1;
    }
}

__device__ inline void deinterleave_words_4bit(
    const uint32_t* const __restrict__ input,
    uint32_t* const __restrict__ output,
    uint32_t bits)
{
    #pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t result_0 = 0, result_1 = 0, result_2 = 0, result_3 = 0;
        #pragma unroll
        for (uint32_t j = 0; j < bits; j += 4) {
            result_0 |= ((input[j] >> (i*4)) & 0xF) << j;
            result_1 |= ((input[j+1] >> (i*4)) & 0xF) << j;
            result_2 |= ((input[j+2] >> (i*4)) & 0xF) << j;
            result_3 |= ((input[j+3] >> (i*4)) & 0xF) << j;
        }
        output[i] = result_0;
        output[i + 8] = result_1;
        output[i + 16] = result_2;
        output[i + 24] = result_3;
    }
}

__device__ inline void deinterleave_words_8bit(
    const uint32_t* const __restrict__ input,
    uint32_t* const __restrict__ output,
    uint32_t bits)
{
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        uint32_t result_0 = 0, result_1 = 0, result_2 = 0, result_3 = 0;
        uint32_t result_4 = 0, result_5 = 0, result_6 = 0, result_7 = 0;
        #pragma unroll
        for (uint32_t j = 0; j < bits; j += 8) {
            result_0 |= ((input[j] >> (i*8)) & 0xFF) << j;
            result_1 |= ((input[j+1] >> (i*8)) & 0xFF) << j;
            result_2 |= ((input[j+2] >> (i*8)) & 0xFF) << j;
            result_3 |= ((input[j+3] >> (i*8)) & 0xFF) << j;
            result_4 |= ((input[j+4] >> (i*8)) & 0xFF) << j;
            result_5 |= ((input[j+5] >> (i*8)) & 0xFF) << j;
            result_6 |= ((input[j+6] >> (i*8)) & 0xFF) << j;
            result_7 |= ((input[j+7] >> (i*8)) & 0xFF) << j;
        }
        output[i] = result_0;
        output[i + 4] = result_1;
        output[i + 8] = result_2;
        output[i + 12] = result_3;
        output[i + 16] = result_4;
        output[i + 20] = result_5;
        output[i + 24] = result_6;
        output[i + 28] = result_7;
    }
}


//------------------------------------------------------------------------------
// GPU De-interleave Kernel Wrappers

__global__ void deinterleave_kernel_1bit(
    const uint32_t* input,
    uint32_t* output,
    uint32_t bits)
{
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    input += thread_idx * bits;
    output += thread_idx * 32;

    // Copy input to local register array
    uint32_t words[32];
    #pragma unroll
    for (uint32_t i = 0; i < bits; ++i) {
        words[i] = input[i];
    }

    deinterleave_words_1bit(words, output, bits);
}

__global__ void deinterleave_kernel_2bit(
    const uint32_t* input,
    uint32_t* output,
    uint32_t bits)
{
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    input += thread_idx * bits;
    output += thread_idx * 32;

    // Copy input to local register array
    uint32_t words[32];
    #pragma unroll
    for (uint32_t i = 0; i < bits; ++i) {
        words[i] = input[i];
    }

    deinterleave_words_2bit(words, output, bits);
}

__global__ void deinterleave_kernel_4bit(
    const uint32_t* input,
    uint32_t* output,
    uint32_t bits)
{
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    input += thread_idx * bits;
    output += thread_idx * 32;

    // Copy input to local register array
    uint32_t words[32];
    #pragma unroll
    for (uint32_t i = 0; i < bits; ++i) {
        words[i] = input[i];
    }

    deinterleave_words_4bit(words, output, bits);
}

__global__ void deinterleave_kernel_8bit(
    const uint32_t* input,
    uint32_t* output,
    uint32_t bits)
{
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    input += thread_idx * bits;
    output += thread_idx * 32;

    // Copy input to local register array
    uint32_t words[32];
    #pragma unroll
    for (uint32_t i = 0; i < bits; ++i) {
        words[i] = input[i];
    }

    deinterleave_words_8bit(words, output, bits);
}


//------------------------------------------------------------------------------
// CPU De-interleave Functions

void cpu_deinterleave_1bit(const uint32_t* input, uint32_t* output, int block_count, int low_bits) {
    for (int block = 0; block < block_count; ++block) {
        const uint32_t* block_input = input + (block * low_bits);
        uint32_t* block_output = output + (block * 32);
        
        for (int i = 0; i < 32; ++i) {
            uint32_t result = 0;
            for (int j = 0; j < low_bits; ++j) {
                result |= ((block_input[j] >> i) & 1) << j;
            }
            block_output[i] = result;
        }
    }
}

void cpu_deinterleave_2bit(const uint32_t* input, uint32_t* output, int block_count, int low_bits) {
    for (int block = 0; block < block_count; ++block) {
        const uint32_t* block_input = input + (block * low_bits);
        uint32_t* block_output = output + (block * 32);
        
        for (int i = 0; i < 16; ++i) {
            uint32_t result_0 = 0, result_1 = 0;
            for (int j = 0; j < low_bits; j += 2) {
                result_0 |= ((block_input[j] >> (i*2)) & 3) << j;
                result_1 |= ((block_input[j+1] >> (i*2)) & 3) << j;
            }
            block_output[i] = result_0;
            block_output[i + 16] = result_1;
        }
    }
}

void cpu_deinterleave_4bit(const uint32_t* input, uint32_t* output, int block_count, int low_bits) {
    for (int block = 0; block < block_count; ++block) {
        const uint32_t* block_input = input + (block * low_bits);
        uint32_t* block_output = output + (block * 32);
        
        for (int i = 0; i < 8; ++i) {
            uint32_t result_0 = 0, result_1 = 0, result_2 = 0, result_3 = 0;
            for (int j = 0; j < low_bits; j += 4) {
                result_0 |= ((block_input[j] >> (i*4)) & 0xF) << j;
                result_1 |= ((block_input[j+1] >> (i*4)) & 0xF) << j;
                result_2 |= ((block_input[j+2] >> (i*4)) & 0xF) << j;
                result_3 |= ((block_input[j+3] >> (i*4)) & 0xF) << j;
            }
            block_output[i] = result_0;
            block_output[i + 8] = result_1;
            block_output[i + 16] = result_2;
            block_output[i + 24] = result_3;
        }
    }
}

void cpu_deinterleave_8bit(const uint32_t* input, uint32_t* output, int block_count, int low_bits) {
    for (int block = 0; block < block_count; ++block) {
        const uint32_t* block_input = input + (block * low_bits);
        uint32_t* block_output = output + (block * 32);

        for (int i = 0; i < 4; ++i) {
            uint32_t result_0 = 0, result_1 = 0, result_2 = 0, result_3 = 0;
            uint32_t result_4 = 0, result_5 = 0, result_6 = 0, result_7 = 0;
            for (int j = 0; j < low_bits; j += 8) {
                result_0 |= ((block_input[j] >> (i*8)) & 0xFF) << j;
                result_1 |= ((block_input[j+1] >> (i*8)) & 0xFF) << j;
                result_2 |= ((block_input[j+2] >> (i*8)) & 0xFF) << j;
                result_3 |= ((block_input[j+3] >> (i*8)) & 0xFF) << j;
                result_4 |= ((block_input[j+4] >> (i*8)) & 0xFF) << j;
                result_5 |= ((block_input[j+5] >> (i*8)) & 0xFF) << j;
                result_6 |= ((block_input[j+6] >> (i*8)) & 0xFF) << j;
                result_7 |= ((block_input[j+7] >> (i*8)) & 0xFF) << j;
            }
            block_output[i] = result_0;
            block_output[i + 4] = result_1;
            block_output[i + 8] = result_2;
            block_output[i + 12] = result_3;
            block_output[i + 16] = result_4;
            block_output[i + 20] = result_5;
            block_output[i + 24] = result_6;
            block_output[i + 28] = result_7;
        }
    }
}


//------------------------------------------------------------------------------
// Test function for de-interleave

void run_deinterleave_test(int bits, int low_bits) {
    const int value_count = 2 * 1024 * 1024;
    const int interleaved_count = 32;
    const int operation_count = value_count / interleaved_count;
    const int block_size = 256;
    const int block_count = operation_count / block_size;

    // Allocate device memory
    uint32_t *d_input, *d_output;
    cudaMallocManaged(&d_input, value_count * sizeof(uint32_t) * low_bits / 32);
    CallbackScope input_cleanup([&]() { cudaFree(d_input); });
    cudaMallocManaged(&d_output, value_count * sizeof(uint32_t));
    CallbackScope output_cleanup([&]() { cudaFree(d_output); });
    cudaCheckError();

    uint32_t mask = generate_mask(low_bits);
    for (int i = 0; i < value_count * low_bits / 32; i++) {
        d_input[i] = rand() & mask;
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
    if (low_bits != 32 && low_bits != 16) {
        num_runs = 1;
    }

    for (int run = 0; run < num_runs; run++) {
        // Run the kernel
        cudaEventRecord(start);
        if (bits == 1) {
            deinterleave_kernel_1bit<<<block_count, block_size>>>(d_input, d_output, low_bits);
        } else if (bits == 2) {
            deinterleave_kernel_2bit<<<block_count, block_size>>>(d_input, d_output, low_bits);
        } else if (bits == 4) {
            deinterleave_kernel_4bit<<<block_count, block_size>>>(d_input, d_output, low_bits);
        } else if (bits == 8) {
            deinterleave_kernel_8bit<<<block_count, block_size>>>(d_input, d_output, low_bits);
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
            cpu_deinterleave_1bit(d_input, expected_output.data(), operation_count, low_bits);
        } else if (bits == 2) {
            cpu_deinterleave_2bit(d_input, expected_output.data(), operation_count, low_bits);
        } else if (bits == 4) {
            cpu_deinterleave_4bit(d_input, expected_output.data(), operation_count, low_bits);
        } else if (bits == 8) {
            cpu_deinterleave_8bit(d_input, expected_output.data(), operation_count, low_bits);
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
        cpu_times.push_back(cpu_duration.count() / 1000.0);

        // Verify results (only for the first run)
        if (run == 0) {
            for (int i = 0; i < value_count; i++) {
                if (d_output[i] != expected_output[i]) {
                    cerr << "Mismatch at index " << i << " (" << bits << " bits de-interleave, low_bits=" << low_bits << "):\n";
                    cerr << "Expected: 0x" << std::hex << expected_output[i] << std::dec << endl;
                    cerr << "Got:      0x" << std::hex << d_output[i] << std::dec << endl;
                    assert(false);
                }
            }
        }
    }

    double median_gpu_time = median(gpu_times);
    double median_cpu_time = median(cpu_times);

    cout << "Test passed for " << bits << "-bit de-interleave with " << value_count << " values, low_bits=" << low_bits << endl;
    cout << "Median GPU Time: " << std::fixed << std::setprecision(3) << median_gpu_time << " ms" << endl;
    cout << "Median CPU Time: " << std::fixed << std::setprecision(3) << median_cpu_time << " ms" << endl;
    cout << "GPU Speedup: " << std::fixed << std::setprecision(2) << (median_cpu_time / median_gpu_time) << "x" << endl;
    cout << endl;
}


//------------------------------------------------------------------------------
// Unit Tests

static void verify_data(const uint32_t* original, const uint32_t* result, int size, const string& test_name) {
    for (int i = 0; i < size; ++i) {
        if (original[i] != result[i]) {
            cerr << "Mismatch in " << test_name << " at index " << i << ":\n";
            cerr << "Original: 0x" << std::hex << original[i] << std::dec << endl;
            cerr << "Result:   0x" << std::hex << result[i] << std::dec << endl;
            assert(false);
        }
    }
    cout << test_name << " passed." << endl;
}

void run_unit_tests(int bits, int low_bits) {
    const int value_count = 1024; // Smaller size for unit tests
    const int interleaved_count = 32;
    const int operation_count = value_count / interleaved_count;
    const int block_size = 32;
    const int block_count = operation_count / block_size;

    // Allocate device memory
    uint32_t *d_original, *d_interleaved, *d_deinterleaved;
    cudaMallocManaged(&d_original, value_count * sizeof(uint32_t));
    cudaMallocManaged(&d_interleaved, value_count * sizeof(uint32_t) * low_bits / 32);
    cudaMallocManaged(&d_deinterleaved, value_count * sizeof(uint32_t));

    // Initialize data
    uint32_t mask = generate_mask(low_bits);
    for (int i = 0; i < value_count; i++) {
        d_original[i] = rand() & mask;
    }

    // CPU interleave followed by GPU de-interleave
    if (bits == 1) {
        cpu_interleave_1bit(d_original, d_interleaved, operation_count, low_bits);
        deinterleave_kernel_1bit<<<block_count, block_size>>>(d_interleaved, d_deinterleaved, low_bits);
    } else if (bits == 2) {
        cpu_interleave_2bit(d_original, d_interleaved, operation_count, low_bits);
        deinterleave_kernel_2bit<<<block_count, block_size>>>(d_interleaved, d_deinterleaved, low_bits);
    } else if (bits == 4) {
        cpu_interleave_4bit(d_original, d_interleaved, operation_count, low_bits);
        deinterleave_kernel_4bit<<<block_count, block_size>>>(d_interleaved, d_deinterleaved, low_bits);
    } else if (bits == 8) {
        cpu_interleave_8bit(d_original, d_interleaved, operation_count, low_bits);
        deinterleave_kernel_8bit<<<block_count, block_size>>>(d_interleaved, d_deinterleaved, low_bits);
    }
    cudaDeviceSynchronize();
    verify_data(d_original, d_deinterleaved, value_count, "CPU interleave + GPU de-interleave (" + to_string(bits) + "-bit, low_bits=" + to_string(low_bits) + ")");

    // GPU interleave followed by CPU de-interleave
    if (bits == 1) {
        interleave_kernel_1bit<<<block_count, block_size>>>(d_original, d_interleaved, low_bits);
        cudaDeviceSynchronize();
        cpu_deinterleave_1bit(d_interleaved, d_deinterleaved, operation_count, low_bits);
    } else if (bits == 2) {
        interleave_kernel_2bit<<<block_count, block_size>>>(d_original, d_interleaved, low_bits);
        cudaDeviceSynchronize();
        cpu_deinterleave_2bit(d_interleaved, d_deinterleaved, operation_count, low_bits);
    } else if (bits == 4) {
        interleave_kernel_4bit<<<block_count, block_size>>>(d_original, d_interleaved, low_bits);
        cudaDeviceSynchronize();
        cpu_deinterleave_4bit(d_interleaved, d_deinterleaved, operation_count, low_bits);
    } else if (bits == 8) {
        interleave_kernel_8bit<<<block_count, block_size>>>(d_original, d_interleaved, low_bits);
        cudaDeviceSynchronize();
        cpu_deinterleave_8bit(d_interleaved, d_deinterleaved, operation_count, low_bits);
    }
    verify_data(d_original, d_deinterleaved, value_count, "GPU interleave + CPU de-interleave (" + to_string(bits) + "-bit, low_bits=" + to_string(low_bits) + ")");

    // GPU interleave followed by GPU de-interleave
    if (bits == 1) {
        interleave_kernel_1bit<<<block_count, block_size>>>(d_original, d_interleaved, low_bits);
        deinterleave_kernel_1bit<<<block_count, block_size>>>(d_interleaved, d_deinterleaved, low_bits);
    } else if (bits == 2) {
        interleave_kernel_2bit<<<block_count, block_size>>>(d_original, d_interleaved, low_bits);
        deinterleave_kernel_2bit<<<block_count, block_size>>>(d_interleaved, d_deinterleaved, low_bits);
    } else if (bits == 4) {
        interleave_kernel_4bit<<<block_count, block_size>>>(d_original, d_interleaved, low_bits);
        deinterleave_kernel_4bit<<<block_count, block_size>>>(d_interleaved, d_deinterleaved, low_bits);
    } else if (bits == 8) {
        interleave_kernel_8bit<<<block_count, block_size>>>(d_original, d_interleaved, low_bits);
        deinterleave_kernel_8bit<<<block_count, block_size>>>(d_interleaved, d_deinterleaved, low_bits);
    }
    cudaDeviceSynchronize();
    verify_data(d_original, d_deinterleaved, value_count, "GPU interleave + GPU de-interleave (" + to_string(bits) + "-bit, low_bits=" + to_string(low_bits) + ")");

    // CPU interleave followed by CPU de-interleave
    if (bits == 1) {
        cpu_interleave_1bit(d_original, d_interleaved, operation_count, low_bits);
        cpu_deinterleave_1bit(d_interleaved, d_deinterleaved, operation_count, low_bits);
    } else if (bits == 2) {
        cpu_interleave_2bit(d_original, d_interleaved, operation_count, low_bits);
        cpu_deinterleave_2bit(d_interleaved, d_deinterleaved, operation_count, low_bits);
    } else if (bits == 4) {
        cpu_interleave_4bit(d_original, d_interleaved, operation_count, low_bits);
        cpu_deinterleave_4bit(d_interleaved, d_deinterleaved, operation_count, low_bits);
    } else if (bits == 8) {
        cpu_interleave_8bit(d_original, d_interleaved, operation_count, low_bits);
        cpu_deinterleave_8bit(d_interleaved, d_deinterleaved, operation_count, low_bits);
    }
    verify_data(d_original, d_deinterleaved, value_count, "CPU interleave + CPU de-interleave (" + to_string(bits) + "-bit, low_bits=" + to_string(low_bits) + ")");

    // Clean up
    cudaFree(d_original);
    cudaFree(d_interleaved);
    cudaFree(d_deinterleaved);
}


int main() {
    cout << "Running unit tests:" << endl;
    for (int low_bits = 32; low_bits >= 1; low_bits--) {
        run_unit_tests(1, low_bits);
        if (low_bits % 2 == 0) {
            run_unit_tests(2, low_bits);
        }
        if (low_bits % 4 == 0) {
            run_unit_tests(4, low_bits);
        }
        if (low_bits % 8 == 0) {
            run_unit_tests(8, low_bits);
        }
    }

    cout << "Running tests " << NUM_RUNS << " times each and reporting median times:" << endl << endl;
    for (int low_bits = 32; low_bits >= 1; low_bits--) {
        run_test(1, low_bits);
        if (low_bits % 2 == 0) {
            run_test(2, low_bits);
        }
        if (low_bits % 4 == 0) {
            run_test(4, low_bits);
        }
        if (low_bits % 8 == 0) {
            run_test(8, low_bits);
        }
    }

    cout << "Running de-interleave tests " << NUM_RUNS << " times each and reporting median times:" << endl << endl;
    for (int low_bits = 32; low_bits >= 1; low_bits--) {
        run_deinterleave_test(1, low_bits);
        if (low_bits % 2 == 0) {
            run_deinterleave_test(2, low_bits);
        }
        if (low_bits % 4 == 0) {
            run_deinterleave_test(4, low_bits);
        }
        if (low_bits % 8 == 0) {
            run_deinterleave_test(8, low_bits);
        }
    }

    cout << "All tests passed!" << endl;
    return 0;
}

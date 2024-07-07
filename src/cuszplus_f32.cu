#include "cuszplus_f32.h"

#include <zstd.h>

#include <cub/cub.cuh> // CUB from CUDA Toolkit

#include <iostream>
using namespace std;


//------------------------------------------------------------------------------
// Constants

static const int kZstdCompressionLevel = 1;
static const uint32_t kMagic = 0xCA7DD007;
static const uint32_t kHeaderBytes = 4 + 4 + 4;

#define BLOCK_SIZE 256
#define QUANT_GROUP_SIZE 32
#define THREAD_GROUP_COUNT 4
#define THREAD_FLOAT_COUNT (THREAD_GROUP_COUNT * QUANT_GROUP_SIZE)
#define BLOCK_FLOAT_COUNT (BLOCK_SIZE * THREAD_FLOAT_COUNT)
#define BLOCK_PARAM_COUNT (BLOCK_SIZE * THREAD_GROUP_COUNT)
#define PARAM_SIZE (4 + 1 + 1)
#define BLOCK_BYTES (BLOCK_PARAM_COUNT*PARAM_SIZE + BLOCK_FLOAT_COUNT*sizeof(float))
#define INTERLEAVE_BITS 4

/*
    Header:
        kMagic(4 bytes)
        Epsilon(4 bytes)
        FloatCount(4 bytes)
        Block 0 used words(4 bytes)
        Block 1 used words(4 bytes)
        ...
        Block N used words(4 bytes)

    Followed by each block:
        MaxIndex(1 byte) x BLOCK_PARAM_COUNT
        Bits(1 byte) x BLOCK_PARAM_COUNT
        HighBits(4 bytes) x BLOCK_PARAM_COUNT
        <Compressed Floats>
            Quantization Group 0(QUANT_GROUP_SIZE * Bits_0 / 8 bytes)
            Quantization Group 1(QUANT_GROUP_SIZE * Bits_1 / 8 bytes)
            ...
            Quantization Group i(QUANT_GROUP_SIZE * Bits_i / 8 bytes)

    Compression Algorithm:
        First quantize (int32_t) each float by dividing by epsilon:
            X[i] = Torch.Round( Float[i] / epsilon )

        Within each GPU thread, subtract consective floats
        for sets of THREAD_GROUP_COUNT * QUANT_GROUP_SIZE floats:
            X[i] = X[i] - X[i - 1]

        Zig-zag encoding: (x << 1) ^ (x >> 31)
        This puts the sign bit in the least significant bit, so that
        each quantized value becomes an unsigned integer.
            X[i] = ZigZagEncode( X[i] )

        Find the two largest values in each QUANT_GROUP_SIZE.
            X_max = Max(X[i]), X_max2 = SecondLargest(X[i])

        Get the number of bits required to represent X_max2.
            Bits = BitCount(X_max2)

        Store index of X_max and the high bits (X_max >> Bits) and the
        number of Bits in a table, for use in decompression.

        Interleave Bits from sets of quantized values.
        So every 32 bits of output corresponds to one vertical column of bits
        from QUANT_GROUP_SIZE quantized values.
        These values are packed together for each block of BLOCK_SIZE threads
        using synchronization between threads in the block.

    The above algorithm runs on GPU massively in parallel.
    This prepares the data for further compression on CPU using Zstd. 
    The result of GPU compression is a set of disjoint blocks.
    Each block is compressed using ZSTD_compressStream as if they were
    contiguous in memory rather than disjoint blocks.

    The header includes all the information needed to decompress the data.


    Decompression Algorithm:

    We decompress the data into a large contiguous buffer that is shared
    with the GPU.  On GPU:

        For each quantization group:
            Unpack the quantized values from the bit packing.

            Restore the largest value from the table.

            X[i] = ZigZagDecode( X[i] ), now a 32-bit signed integer.

            X[i] = X[i] + X[i - 1]

            X[i] = X[i] * epsilon

    The result will be the original set of floating point numbers that can be
    read back to the CPU.


    Discussion:

    I believe this algorithm is a good compromise between speed, compression
    ratio, and simplicity.

    Floating-point values are quantized to a given epsilon.  This is done by
    dividing by epsilon and rounding to the nearest integer.
    Torch.Round style rounding is used to improve the quality of the rounding,
    where it rounds towards the nearest even value (including 0).
    The remaining operations are all lossless.

    Subtracting subsequent values is a simple predictor, which is just one
    option.  For example it could subtract future values or vertically adjacent
    values.  However, all of these predictors are pretty similar in performance
    and this is the most efficient option.  To improve further, the compressor
    could consider alternative predictors and store the best one, but it would
    slow down the algorithm and add a significant amount of complexity.
    There's an example of this more complex algorithm here:
    https://github.com/catid/Zdepth/blob/ac7c6d8e944d07be2404e5a1eaa04562595f3756/src/zdepth.cpp#L437 

    Zig-Zag encoding is used, as described in https://lemire.me/blog/2022/11/25/making-all-your-integers-positive-with-zigzag-encoding/
    This is a simple way to encode signed integers into unsigned integers,
    so that the high bits are all zeros.

    The exception list of a single largest value per quantization group is
    inspired by https://github.com/lemire/FastPFor
    specifically this part: https://github.com/lemire/FastPFor/blob/8ba17c93ed2c173caba839427858a16236833f77/headers/pfor.h#L306
    To make it GPU friendly and simpler, this algorithm always has just one
    exception per quantization group.  To motivate this, consider the case
    where the values are similar to eachother but large.  In this case, the
    first value will be much larger than the rest after subtracting neighbors.
    If we make that first value an exception, then the rest of the values can
    be packed together much more tightly.

    Further improvements might be achieved by interleaving multiple blocks
    in a second pass through the data prior to Zstd compression, though I
    suspect this would have diminishing returns.
*/


//------------------------------------------------------------------------------
// Serialization

union FloatUInt32 {
    float f;
    uint32_t u;
};

inline void write_uint32_le(void* buffer, uint32_t value) {
    uint8_t* ptr = static_cast<uint8_t*>(buffer);
    ptr[0] = static_cast<uint8_t>(value);
    ptr[1] = static_cast<uint8_t>(value >> 8);
    ptr[2] = static_cast<uint8_t>(value >> 16);
    ptr[3] = static_cast<uint8_t>(value >> 24);
}

inline uint32_t read_uint32_le(const void* buffer) {
    const uint8_t* ptr = static_cast<const uint8_t*>(buffer);
    return static_cast<uint32_t>(ptr[0]) |
           (static_cast<uint32_t>(ptr[1]) << 8) |
           (static_cast<uint32_t>(ptr[2]) << 16) |
           (static_cast<uint32_t>(ptr[3]) << 24);
}

inline void write_float_le(void* buffer, float value) {
    FloatUInt32 conv;
    conv.f = value;
    write_uint32_le(buffer, conv.u);
}

inline float read_float_le(const void* buffer) {
    FloatUInt32 conv;
    conv.u = read_uint32_le(buffer);
    return conv.f;
}


//------------------------------------------------------------------------------
// Tools

struct CallbackScope {
    CallbackScope(std::function<void()> func) : func(func) {}
    ~CallbackScope() { func(); }
    std::function<void()> func;
};

__device__ inline uint32_t bit_count(uint32_t x)
{
    return (sizeof(uint32_t)*8) - __clz(x);
}

__device__ inline uint32_t zigzag_encode(uint32_t x)
{
    return (x << 1) ^ (x >> 31);
}

__device__ inline int32_t zigzag_decode(uint32_t x)
{
    return (x >> 1) ^ -(x & 1);
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
// CPU De-interleave Functions

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


//------------------------------------------------------------------------------
// Compression Kernel

__global__ void SZplus_compress(
    const float* const __restrict__ original_data,
    float epsilon,
    uint32_t* __restrict__ block_used_words,
    uint8_t* __restrict__ compressed_data)
{
    using BlockScan = cub::BlockScan<uint32_t, BLOCK_SIZE>;
    using BlockAdjacentDifferenceT = cub::BlockAdjacentDifference<int32_t, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    compressed_data += blockIdx.x * BLOCK_BYTES;

    uint32_t quant_group[THREAD_FLOAT_COUNT];
    uint8_t group_bits[THREAD_GROUP_COUNT];

    epsilon = 1.0f / epsilon;
    uint32_t used_words = 0;
    int32_t prev_quant = 0;
    for (int i = 0; i < THREAD_GROUP_COUNT; i++) {
        uint32_t max_quant = 0, max2_quant = 0;
        uint8_t max_index = 0;

        for (int j = 0; j < QUANT_GROUP_SIZE; j++) {
            int float_index = thread_idx * THREAD_FLOAT_COUNT + j;
            float f = original_data[float_index] * epsilon;

            // This is the same quantization used by torch.round()
            const int32_t quant = __float2int_rn(f);

            const int32_t delta = quant - prev_quant;
            prev_quant = quant;

            const uint32_t zig_quant = zigzag_encode(delta);
            quant_group[i * QUANT_GROUP_SIZE + j] = zig_quant;

            // Update max_quant and max2_quant
            if (zig_quant > max_quant) {
                max2_quant = max_quant;
                max_quant = zig_quant;
                max_index = (uint8_t)j;
            } else if (zig_quant > max2_quant) {
                max2_quant = zig_quant;
            }
        }

        // Number of bits to represent second largest and smaller quantized values
        const uint32_t bits = bit_count(max2_quant);

        // Increment write count for this quantization group
        used_words += bits * QUANT_GROUP_SIZE / sizeof(uint32_t);

        group_bits[i] = static_cast<uint8_t>(bits);

        const uint32_t max_high_bits = (bits >= 32) ? 0 : (max_quant >> bits);

        // For each QUANT_GROUP_SIZE, write the number of bits, index of max value, and high bits
        uint8_t* __restrict__ params = compressed_data + BLOCK_PARAM_COUNT * threadIdx.x + i;
        params[0] = static_cast<uint8_t>(max_index);
        params[BLOCK_PARAM_COUNT] = static_cast<uint8_t>(bits);
        params[BLOCK_PARAM_COUNT*2] = static_cast<uint8_t>(max_high_bits);
        params[BLOCK_PARAM_COUNT*3] = static_cast<uint8_t>(max_high_bits >> 8);
        params[BLOCK_PARAM_COUNT*4] = static_cast<uint8_t>(max_high_bits >> 16);
        params[BLOCK_PARAM_COUNT*5] = static_cast<uint8_t>(max_high_bits >> 24);
    }

    __syncthreads(); // Barrier for smem reuse

    // Inclusive Sum (using CUB)
    BlockScan block_scan(temp_storage);
    uint32_t offset = 0;
    block_scan.ExclusiveSum(used_words, offset);

    __syncthreads(); // Barrier for smem reuse

    if (threadIdx.x == blockDim.x - 1) {
        block_used_words[blockIdx.x] = offset + used_words;
    }

    // Get pointer to compressed words for this thread
    compressed_data += BLOCK_PARAM_COUNT * PARAM_SIZE;
    uint32_t* __restrict__ compressed_words = reinterpret_cast<uint32_t*>(compressed_data);
    compressed_words += offset;

    for (int i = 0; i < THREAD_GROUP_COUNT; i++) {
        const uint32_t bits = group_bits[i];

#if INTERLEAVE_BITS == 1
        interleave_words_1bit(quant_group + i * QUANT_GROUP_SIZE, compressed_words, bits);
#elif INTERLEAVE_BITS == 2
        interleave_words_2bit(quant_group + i * QUANT_GROUP_SIZE, compressed_words, bits);
#elif INTERLEAVE_BITS == 4
        interleave_words_4bit(quant_group + i * QUANT_GROUP_SIZE, compressed_words, bits);
#elif INTERLEAVE_BITS == 8
        interleave_words_8bit(quant_group + i * QUANT_GROUP_SIZE, compressed_words, bits);
#else
#error "Invalid INTERLEAVE_BITS value. Must be 1, 2, 4, or 8."
#endif

        compressed_words += bits;
    }
}


//------------------------------------------------------------------------------
// Decompression Kernel

__global__ void SZplus_decompress(
    float* const __restrict__ decompressed_floats,
    const uint8_t* __restrict__ compressed_blocks,
    const uint32_t* const __restrict__ block_used_words,
    float epsilon)
{
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_float_start = thread_idx * THREAD_FLOAT_COUNT;

    // FIXME: Fix the block calculation here - Should be using offset list

    const uint8_t* block_data = compressed_blocks + blockIdx.x * (BLOCK_PARAM_COUNT * PARAM_SIZE + BLOCK_FLOAT_COUNT * sizeof(float));
    const uint32_t* compressed_words = reinterpret_cast<const uint32_t*>(block_data + BLOCK_PARAM_COUNT * PARAM_SIZE);

    uint32_t quant_group[THREAD_FLOAT_COUNT];
    int32_t prev_quant = 0;

    for (int i = 0; i < THREAD_GROUP_COUNT; i++) {
        const uint8_t max_index = block_data[THREAD_GROUP_COUNT * threadIdx.x + i];
        const uint8_t bits = block_data[BLOCK_PARAM_COUNT + THREAD_GROUP_COUNT * threadIdx.x + i];
        const uint32_t* high_bits = reinterpret_cast<const uint32_t*>(block_data + BLOCK_PARAM_COUNT * 2);
        const uint32_t max_quant = (high_bits[THREAD_GROUP_COUNT * threadIdx.x + i] << bits) | ((1 << bits) - 1);

        if (bits == 0) {
            for (int j = 0; j < QUANT_GROUP_SIZE; j++) {
                quant_group[i * QUANT_GROUP_SIZE + j] = 0;
            }
        } else {
#if INTERLEAVE_BITS == 1
            deinterleave_words_1bit(compressed_words, quant_group + i * QUANT_GROUP_SIZE, bits);
#elif INTERLEAVE_BITS == 2
            deinterleave_words_2bit(compressed_words, quant_group + i * QUANT_GROUP_SIZE, bits);
#elif INTERLEAVE_BITS == 4
            deinterleave_words_4bit(compressed_words, quant_group + i * QUANT_GROUP_SIZE, bits);
#elif INTERLEAVE_BITS == 8
            deinterleave_words_8bit(compressed_words, quant_group + i * QUANT_GROUP_SIZE, bits);
#else
#error "Invalid INTERLEAVE_BITS value. Must be 1, 2, 4, or 8."
#endif
        }

        compressed_words += bits;

        // Restore the max value
        quant_group[i * QUANT_GROUP_SIZE + max_index] = max_quant;

        // Zig-zag decode and inverse delta encoding
        for (int j = 0; j < QUANT_GROUP_SIZE; j++) {
            int32_t curr_quant = zigzag_decode(quant_group[i * QUANT_GROUP_SIZE + j]) + prev_quant;
            prev_quant = curr_quant;

            int float_index = block_float_start + i * QUANT_GROUP_SIZE + j;
            decompressed_floats[float_index] = curr_quant * epsilon;
        }
    }
}


//------------------------------------------------------------------------------
// Compress API

int GetCompressedBufferSize(int float_count)
{
    const int block_count = (float_count + BLOCK_FLOAT_COUNT - 1) / BLOCK_FLOAT_COUNT;
    return block_count * BLOCK_BYTES;
}

bool CompressFloats(
    cudaStream_t stream,
    const float* float_data,
    bool is_device_ptr,
    int float_count,
    uint8_t* compressed_buffer,
    int& compressed_bytes,
    float epsilon)
{
    // If it's not a device pointer, we need to copy the data to the device
    float* d_original_data = nullptr;
    if (!is_device_ptr) {
        cudaError_t err = cudaMalloc((void**)&d_original_data, sizeof(float)*float_count);
        if (err != cudaSuccess) {
            cerr << "cudaMalloc failed: err=" << cudaGetErrorString(err) << " float_count=" << float_count << endl;
            return false;
        }
        cudaMemcpyAsync(d_original_data, float_data, sizeof(float)*float_count, cudaMemcpyHostToDevice, stream);
        float_data = d_original_data;
    }
    CallbackScope original_data_cleanup([&]() { if (d_original_data) cudaFree(d_original_data); });

    const int block_count = (float_count + BLOCK_FLOAT_COUNT - 1) / BLOCK_FLOAT_COUNT;

    // Create output buffer
    uint8_t* compressed_blocks = nullptr;
    cudaError_t err = cudaMallocAsync((void**)&compressed_blocks, block_count*BLOCK_BYTES + block_count*sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        cerr << "cudaMallocAsync failed: err=" << cudaGetErrorString(err) << " block_count=" << block_count << endl;
        return false;
    }
    CallbackScope original_comp_cleanup([&]() { cudaFreeAsync(compressed_blocks, stream); });
    uint32_t* block_used_words = reinterpret_cast<uint32_t*>(compressed_blocks + block_count*BLOCK_BYTES);

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(block_count);
    SZplus_compress<<<gridSize, blockSize, 0, stream>>>(
        float_data,
        epsilon,
        block_used_words,
        compressed_blocks);

    // Initialize Zstd
    auto zcs = ZSTD_createCStream();
    if (!zcs) {
        cerr << "ERROR: ZSTD_createCStream failed" << endl;
        return false;
    }
    CallbackScope zcs_cleanup([&]() { ZSTD_freeCStream(zcs); });

    // Initialize the compression stream
    size_t const initResult = ZSTD_initCStream(zcs, kZstdCompressionLevel);
    if (ZSTD_isError(initResult)) {
        cerr << "ERROR: ZSTD_initCStream failed: initResult=" << initResult << endl;
        return false;
    }

    ZSTD_outBuffer output = { compressed_buffer, (unsigned)compressed_bytes, 0 };

    // Prepare header
    std::vector<uint8_t> header_buffer(kHeaderBytes + block_count*sizeof(uint32_t));
    write_uint32_le(header_buffer.data(), kMagic);
    write_float_le(header_buffer.data() + 4, epsilon);
    write_uint32_le(header_buffer.data() + 8, float_count);
    uint32_t* header_words = reinterpret_cast<uint32_t*>(header_buffer.data() + kHeaderBytes);

    cudaStreamSynchronize(stream);
    if (cudaSuccess != cudaGetLastError()) {
        cerr << "ERROR: Encountered a CUDA error during FloatCompressor::Compress" << endl;
        return false;
    }

    // Interleave block used words into header
    cpu_interleave_4bit(block_used_words, header_words, block_count, 32);

    // Write header
    ZSTD_inBuffer input = { header_buffer.data(), header_buffer.size(), 0 };
    while (input.pos < input.size) {
        size_t const compressResult = ZSTD_compressStream(zcs, &output, &input);
        if (ZSTD_isError(compressResult)) {
            cerr << "ERROR: ZSTD_compressStream failed: compressResult=" << compressResult << endl;
            return false;
        }
    }

    // For each compressed block:
    for (int i = 0; i < block_count; i++)
    {
        const uint32_t used_words = block_used_words[i];
        const uint8_t* block_data = compressed_blocks + BLOCK_BYTES * i;
        const uint32_t block_used_bytes = BLOCK_PARAM_COUNT*PARAM_SIZE + used_words * 4;

        ZSTD_inBuffer input = { block_data, block_used_bytes, 0 };
        while (input.pos < input.size) {
            size_t const compressResult = ZSTD_compressStream(zcs, &output, &input);
            if (ZSTD_isError(compressResult)) {
                cerr << "ERROR: ZSTD_compressStream failed: compressResult=" << compressResult << endl;
                return false;
            }

            if (output.pos == output.size) {
                cerr << "ERROR: ZSTD_compressStream failed: compressResult=" << compressResult << endl;
                return false;
            }
        }
    }

    // Flush the zstd stream
    size_t remainingToFlush;
    for (;;) {
        remainingToFlush = ZSTD_endStream(zcs, &output);
        if (ZSTD_isError(remainingToFlush)) {
            cerr << "ERROR: ZSTD_endStream failed: remainingToFlush=" << remainingToFlush << endl;
            return false;
        }

        if (remainingToFlush <= 0) {
            break;
        }

        if (output.pos == output.size) {
            cerr << "ERROR: ZSTD_endStream failed: remainingToFlush=" << remainingToFlush << std::endl;
            return false;
        }
    }

    compressed_bytes = static_cast<int>(output.pos);
    return true;
}


//------------------------------------------------------------------------------
// Decompress API

int GetDecompressedFloatCount(
    const void* compressed_data,
    int compressed_bytes)
{
    if (compressed_bytes < kHeaderBytes) {
        cerr << "ERROR: Compressed data is too small. Expected at least " << kHeaderBytes << " bytes, but got " << compressed_bytes << " bytes." << endl;
        return -1;  // Return -1 to indicate an error
    }

    const uint8_t* data = static_cast<const uint8_t*>(compressed_data);

    // Check magic number
    const uint32_t magic = read_uint32_le(data);
    if (magic != kMagic) {
        cerr << "ERROR: Invalid magic number. Expected 0x" << hex << kMagic << ", but got 0x" << magic << dec << endl;
        return -1;  // Return -1 to indicate an error
    }

    return read_uint32_le(data + 8);
}

bool DecompressFloats(
    cudaStream_t stream,
    const void* compressed_data,
    int compressed_bytes,
    float* decompressed_floats)
{
    if (compressed_bytes < kHeaderBytes) {
        cerr << "ERROR: Compressed data is too small. Expected at least " << kHeaderBytes << " bytes, but got " << compressed_bytes << " bytes." << endl;
        return false;
    }

    // Read header
    const uint8_t* header_data = static_cast<const uint8_t*>(compressed_data);
    const uint32_t magic = read_uint32_le(header_data);
    if (magic != kMagic) {
        cerr << "ERROR: Invalid magic number. Expected 0x" << hex << kMagic << ", but got 0x" << magic << dec << endl;
        return false;
    }
    const float epsilon = read_float_le(header_data + 4);
    const int float_count = read_uint32_le(header_data + 8);

    cerr << "INFO: Decompressing " << float_count << " floats with epsilon " << epsilon << endl;

    // Calculate block count
    const int block_count = (float_count + BLOCK_FLOAT_COUNT - 1) / BLOCK_FLOAT_COUNT;

    cerr << "INFO: Block count: " << block_count << ", Block bytes: " << BLOCK_BYTES << endl;

    // Read block used words
    std::vector<uint32_t> block_used_words(block_count);
    const uint32_t* header_words = reinterpret_cast<const uint32_t*>(header_data + kHeaderBytes);
    cpu_deinterleave_4bit(header_words, block_used_words.data(), block_count, 32);

    // Initialize zstd
    ZSTD_DStream* zds = ZSTD_createDStream();
    if (zds == NULL) {
        cerr << "ERROR: Failed to create ZSTD_DStream" << endl;
        return false;
    }
    CallbackScope zds_cleanup([&]() { ZSTD_freeDStream(zds); });

    size_t const init_result = ZSTD_initDStream(zds);
    if (ZSTD_isError(init_result)) {
        cerr << "ERROR: Failed to initialize ZSTD_DStream: " << ZSTD_getErrorName(init_result) << endl;
        return false;
    }

    // Allocate CUDA managed memory for decompressed blocks
    uint8_t* managed_decompressed_blocks = nullptr;
    cudaError_t err = cudaMallocManaged((void**)&managed_decompressed_blocks, block_count * BLOCK_BYTES, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        cerr << "ERROR: Failed to allocate CUDA managed memory for decompressed blocks: " << cudaGetErrorString(err) << endl;
        return false;
    }
    CallbackScope managed_decompressed_blocks_cleanup([&]() { cudaFree(managed_decompressed_blocks); });

    // Decompress each block
    ZSTD_inBuffer input = { compressed_data, static_cast<size_t>(compressed_bytes), kHeaderBytes + block_count * 4 };
    for (int i = 0; i < block_count; ++i) {
        ZSTD_outBuffer output = { managed_decompressed_blocks + i * BLOCK_BYTES, BLOCK_BYTES, 0 };

        // FIXME: Check this!
        while (output.pos < block_used_words[i] * 4 + BLOCK_PARAM_COUNT * PARAM_SIZE) {
            size_t const result = ZSTD_decompressStream(zds, &output, &input);
            if (ZSTD_isError(result)) {
                cerr << "ERROR: ZSTD decompression failed for block " << i << ": " << ZSTD_getErrorName(result) << endl;
                return false;
            }
            if (result == 0) {
                break;
            }
        }
    }

    // Launch decompression kernel
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(block_count);
    cerr << "INFO: Launching decompression kernel with grid size " << gridSize.x << " and block size " << blockSize.x << endl;
    SZplus_decompress<<<gridSize, blockSize, 0, stream>>>(
        decompressed_floats,
        managed_decompressed_blocks,
        block_used_words.data(),
        epsilon);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "ERROR: Decompression kernel launch failed: " << cudaGetErrorString(err) << endl;
        return false;
    }

    // Wait for kernel to finish
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cerr << "ERROR: Decompression kernel execution failed: " << cudaGetErrorString(err) << endl;
        return false;
    }

    cerr << "INFO: Decompression completed successfully" << endl;
    return true;
}

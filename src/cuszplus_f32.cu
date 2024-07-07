#include "cuszplus_f32.h"

#include <zstd.h>

#include <cuda_runtime.h>
#include <cub/cub.cuh> // CUB from CUDA Toolkit

#include <iostream>
using namespace std;


//------------------------------------------------------------------------------
// Constants

static const int kZstdCompressionLevel = 1;
static const uint32_t kMagic = 0xCA7DD007;
static const int kHeaderBytes = 8;

#define BLOCK_SIZE 256
#define QUANT_GROUP_SIZE 32
#define THREAD_GROUP_COUNT 4
#define THREAD_FLOAT_COUNT (THREAD_GROUP_COUNT * QUANT_GROUP_SIZE)
#define BLOCK_FLOAT_COUNT (BLOCK_SIZE * THREAD_FLOAT_COUNT)
#define BLOCK_PARAM_COUNT (BLOCK_SIZE * THREAD_GROUP_COUNT)
#define PARAM_SIZE (4 + 1 + 1)

/*
    Header:
        kMagic(4 bytes)
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
*/


//------------------------------------------------------------------------------
// Serialization

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
// Interleave Kernels

__device__ void interleave_words_1bit(
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

__device__ void interleave_words_2bit(
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

__device__ void interleave_words_4bit(
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

__device__ void interleave_words_8bit(
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

__device__ void deinterleave_words_1bit(
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

__device__ void deinterleave_words_2bit(
    const uint32_t* const __restrict__ input,
    uint32_t* const __restrict__ output,
    uint32_t bits)
{
    #pragma unroll
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t result_0 = 0, result_1 = 0;
        #pragma unroll
        for (uint32_t j = 0; j < 16; j++) {
            result_0 |= ((input[j*2] >> (i*2)) & 3) << (j*2);
            result_1 |= ((input[j*2+1] >> (i*2)) & 3) << (j*2);
        }
        output[i] = result_0;
        output[i + 16] = result_1;
    }
}

__device__ void deinterleave_words_4bit(
    const uint32_t* const __restrict__ input,
    uint32_t* const __restrict__ output,
    uint32_t bits)
{
    #pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t result_0 = 0, result_1 = 0, result_2 = 0, result_3 = 0;
        #pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            result_0 |= ((input[j*4] >> (i*4)) & 0xF) << (j*4);
            result_1 |= ((input[j*4+1] >> (i*4)) & 0xF) << (j*4);
            result_2 |= ((input[j*4+2] >> (i*4)) & 0xF) << (j*4);
            result_3 |= ((input[j*4+3] >> (i*4)) & 0xF) << (j*4);
        }
        output[i] = result_0;
        output[i + 8] = result_1;
        output[i + 16] = result_2;
        output[i + 24] = result_3;
    }
}

__device__ void deinterleave_words_8bit(
    const uint32_t* const __restrict__ input,
    uint32_t* const __restrict__ output,
    uint32_t bits)
{
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        uint32_t result_0 = 0, result_1 = 0, result_2 = 0, result_3 = 0;
        uint32_t result_4 = 0, result_5 = 0, result_6 = 0, result_7 = 0;
        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            result_0 |= ((input[j*8] >> (i*8)) & 0xFF) << (j*8);
            result_1 |= ((input[j*8+1] >> (i*8)) & 0xFF) << (j*8);
            result_2 |= ((input[j*8+2] >> (i*8)) & 0xFF) << (j*8);
            result_3 |= ((input[j*8+3] >> (i*8)) & 0xFF) << (j*8);
            result_4 |= ((input[j*8+4] >> (i*8)) & 0xFF) << (j*8);
            result_5 |= ((input[j*8+5] >> (i*8)) & 0xFF) << (j*8);
            result_6 |= ((input[j*8+6] >> (i*8)) & 0xFF) << (j*8);
            result_7 |= ((input[j*8+7] >> (i*8)) & 0xFF) << (j*8);
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
// Compression Kernel

__global__ void SZplus_compress_kernel_f32(
    const float* const __restrict__ original_data,
    float epsilon,
    uint32_t* __restrict__ block_used_words,
    uint8_t* __restrict__ compressed_data)
{
    using BlockScan = cub::BlockScan<uint32_t, BLOCK_SIZE>;
    using BlockAdjacentDifferenceT = cub::BlockAdjacentDifference<int32_t, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    compressed_data += blockIdx.x * (BLOCK_PARAM_COUNT * PARAM_SIZE + BLOCK_FLOAT_COUNT * sizeof(float));

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

        // For each QUANT_GROUP_SIZE, write the number of bits, index of max value, and high bits
        compressed_data[THREAD_GROUP_COUNT * threadIdx.x + i] = static_cast<uint8_t>(max_index);
        compressed_data[BLOCK_PARAM_COUNT + THREAD_GROUP_COUNT * threadIdx.x + i] = static_cast<uint8_t>(bits);

        uint32_t* __restrict__ compressed_high_bits = reinterpret_cast<uint32_t*>(compressed_data + BLOCK_PARAM_COUNT * 2);
        compressed_high_bits[THREAD_GROUP_COUNT * threadIdx.x + i] = max_quant >> bits;
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

        // TBD: Try other interleave kernels
        interleave_words_1bit(quant_group + i * QUANT_GROUP_SIZE, compressed_words, bits);
        compressed_words += bits;
    }
}


//------------------------------------------------------------------------------
// FloatCompressor

bool FloatCompressor::Compress(
    const float* float_data,
    int float_count,
    float epsilon)
{
    // Copy data to device
    float* original_data = nullptr;
    cudaError_t err = cudaMalloc((void**)&original_data, sizeof(float)*float_count);
    if (err != cudaSuccess) {
        cerr << "cudaMalloc failed: err=" << cudaGetErrorString(err) << " float_count=" << float_count << endl;
        return -1;
    }
    CallbackScope original_data_cleanup([&]() { cudaFree(original_data); });
    cudaMemcpy(original_data, float_data, sizeof(float)*float_count, cudaMemcpyHostToDevice);

    int block_count = (float_count + BLOCK_FLOAT_COUNT - 1) / BLOCK_FLOAT_COUNT;
    const int block_bytes = BLOCK_PARAM_COUNT*PARAM_SIZE + BLOCK_FLOAT_COUNT*sizeof(float);

    // Create output buffer
    uint8_t* compressed_blocks = nullptr;
    cudaError_t err = cudaMallocManaged((void**)&compressed_blocks, block_count*block_bytes + block_count*sizeof(uint32_t));
    if (err != cudaSuccess) {
        cerr << "cudaMallocManaged failed: err=" << cudaGetErrorString(err) << " block_count=" << block_count << endl;
        return -1;
    }
    CallbackScope original_data_cleanup([&]() { cudaFree(compressed_blocks); });
    uint32_t* block_used_words = reinterpret_cast<uint32_t*>(compressed_blocks + block_count*block_bytes);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    CallbackScope stream_cleanup([&]() { cudaStreamDestroy(stream); });

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(block_count);
    SZplus_compress_kernel_f32<<<gridSize, blockSize, 0, stream>>>(
        original_data,
        epsilon,
        block_used_words,
        compressed_blocks);

    // Initialize zstd
    auto zcs = ZSTD_createCStream();
    if (zcs == NULL) { return false; }
    CallbackScope zcs_cleanup([&]() { ZSTD_freeCStream(zcs); });

    // Initialize the compression stream
    size_t const initResult = ZSTD_initCStream(zcs, kZstdCompressionLevel);
    if (ZSTD_isError(initResult)) { return false; }

    Result.resize(block_bytes * block_count);
    ZSTD_outBuffer output = { Result.data(), Result.size(), 0 };

    cudaDeviceSynchronize();

    {
        std::vector<uint8_t> header_buffer(kHeaderBytes + block_count*sizeof(uint32_t));
        write_uint32_le(header_buffer.data(), kMagic);
        write_uint32_le(header_buffer.data() + 4, float_count);
        uint32_t* words = reinterpret_cast<uint32_t*>(header_buffer.data() + kHeaderBytes);
        for (int i = 0; i < block_count; i++) {
            words[i] = block_used_words[i];
        }

        ZSTD_inBuffer input = { header_buffer.data(), header_buffer.size(), 0 };

        while (input.pos < input.size) {
            size_t const compressResult = ZSTD_compressStream(zcs, &output, &input);
            if (ZSTD_isError(compressResult)) { return false; }

            if (output.pos == output.size) {
                return false;
            }
        }
    }

    // For each compressed block:
    for (int i = 0; i < block_count; i++)
    {
        const uint32_t used_words = block_used_words[i];
        const uint8_t* block_data = compressed_blocks + block_bytes * i;
        const uint32_t block_used_bytes = BLOCK_PARAM_COUNT * PARAM_SIZE + used_words * 4;

        ZSTD_inBuffer input = { block_data, block_used_bytes, 0 };

        while (input.pos < input.size) {
            size_t const compressResult = ZSTD_compressStream(zcs, &output, &input);
            if (ZSTD_isError(compressResult)) { return false; }

            if (output.pos == output.size) {
                return false;
            }
        }
    }

    // Flush the zstd stream
    size_t remainingToFlush;
    for (;;) {
        remainingToFlush = ZSTD_endStream(zcs, &output);
        if (ZSTD_isError(remainingToFlush)) { return false; }

        if (remainingToFlush <= 0) {
            break;
        }

        if (output.pos == output.size) {
            return false;
        }
    }

    Result.resize(output.pos);
    return true;
}


//------------------------------------------------------------------------------
// Decompression Kernel

__global__ void SZplus_decompress_kernel_f32(
    float* const __restrict__ decompressed_floats,
    const uint8_t* __restrict__ compressed_blocks,
    volatile uint32_t* const __restrict__ compressed_block_offsets,
    const float epsilon)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const uint32_t block_offset = compressed_block_offsets[blockIdx.x];
    compressed_blocks += block_offset;

    uint32_t absQuant[32];
    int fixed_rate[block_num];
    uint32_t thread_ofs = 0;

    for (int j = 0; j < block_num; j++)
    {
        const int block_idx = start_block_idx + j;
        if (block_idx < rate_ofs)
        {
            const int rate = (int)cmpData[block_idx];
            fixed_rate[j] = rate;
            thread_ofs += rate != 0 ? (32 + rate * 32) : 0; 
        }
    }

    uint32_t cmp_byte_ofs = sync_offsets(thread_ofs, cmpOffset, flag, nbEle);

    for (int j = 0; j < block_num; j++)
    {
        const int temp_start_idx = start_idx + j*32;
        const int rate = fixed_rate[j];

        if (rate)
        {
            uint8_t tmp_char0, tmp_char1, tmp_char2, tmp_char3;
            for(int i=0; i<32; i++) absQuant[i] = 0;
            for(int i=0; i<fixed_rate[j]; i++)
            {
                absQuant[0] |= ((tmp_char0 >> 7) & 0x00000001) << i;
                absQuant[1] |= ((tmp_char0 >> 6) & 0x00000001) << i;
                absQuant[2] |= ((tmp_char0 >> 5) & 0x00000001) << i;
                absQuant[3] |= ((tmp_char0 >> 4) & 0x00000001) << i;
                absQuant[4] |= ((tmp_char0 >> 3) & 0x00000001) << i;
                absQuant[5] |= ((tmp_char0 >> 2) & 0x00000001) << i;
                absQuant[6] |= ((tmp_char0 >> 1) & 0x00000001) << i;
                absQuant[7] |= ((tmp_char0 >> 0) & 0x00000001) << i;

                absQuant[8] |= ((tmp_char1 >> 7) & 0x00000001) << i;
                absQuant[9] |= ((tmp_char1 >> 6) & 0x00000001) << i;
                absQuant[10] |= ((tmp_char1 >> 5) & 0x00000001) << i;
                absQuant[11] |= ((tmp_char1 >> 4) & 0x00000001) << i;
                absQuant[12] |= ((tmp_char1 >> 3) & 0x00000001) << i;
                absQuant[13] |= ((tmp_char1 >> 2) & 0x00000001) << i;
                absQuant[14] |= ((tmp_char1 >> 1) & 0x00000001) << i;
                absQuant[15] |= ((tmp_char1 >> 0) & 0x00000001) << i;

                absQuant[16] |= ((tmp_char2 >> 7) & 0x00000001) << i;
                absQuant[17] |= ((tmp_char2 >> 6) & 0x00000001) << i;
                absQuant[18] |= ((tmp_char2 >> 5) & 0x00000001) << i;
                absQuant[19] |= ((tmp_char2 >> 4) & 0x00000001) << i;
                absQuant[20] |= ((tmp_char2 >> 3) & 0x00000001) << i;
                absQuant[21] |= ((tmp_char2 >> 2) & 0x00000001) << i;
                absQuant[22] |= ((tmp_char2 >> 1) & 0x00000001) << i;
                absQuant[23] |= ((tmp_char2 >> 0) & 0x00000001) << i;

                absQuant[24] |= ((tmp_char3 >> 7) & 0x00000001) << i;
                absQuant[25] |= ((tmp_char3 >> 6) & 0x00000001) << i;
                absQuant[26] |= ((tmp_char3 >> 5) & 0x00000001) << i;
                absQuant[27] |= ((tmp_char3 >> 4) & 0x00000001) << i;
                absQuant[28] |= ((tmp_char3 >> 3) & 0x00000001) << i;
                absQuant[29] |= ((tmp_char3 >> 2) & 0x00000001) << i;
                absQuant[30] |= ((tmp_char3 >> 1) & 0x00000001) << i;
                absQuant[31] |= ((tmp_char3 >> 0) & 0x00000001) << i;
            }

            int32_t prevQuant = 0;
            for (int i = 0; i < 32; i++)
            {
                int32_t currQuant = zigzag_decode(absQuant[i]) + prevQuant;
                decData[temp_start_idx + i] = currQuant * eb;
                prevQuant = currQuant;
            }
        } else {
            for(int i = 0; i < 32; i++)
            {
                decData[temp_start_idx + i] = 0.f;
            }
        }
    }
}


//------------------------------------------------------------------------------
// FloatDecompressor

bool FloatDecompressor::Decompress(
    const void* compressed_data,
    int compressed_bytes)
{
    if (compressed_bytes < kHeaderBytes) {
        return false;
    }



    const size_t r = ZSTD_decompress(
        decompressed,
        original_bytes,
        compressed_data,
        compressed_bytes);

    if (ZSTD_isError(r) || r != (size_t)original_bytes) {
        LOG_ERROR() << "Decompressor: Failed to decompress: r=" << r
            << " original_bytes=" << original_bytes << " err=" << ZSTD_getErrorName(r);
        return false;
    }

    uint32_t* unpacked = reinterpret_cast<uint32_t*>( Result.data() );

    for (uint32_t i = 0; i < original_tokens; ++i) {
        uint8_t unpacked_word[8] = {0};
        for (uint32_t j = 0; j < token_bytes; j++) {
            unpacked_word[j] = decompressed[i + j * original_tokens];
        }

        unpacked[i] = read_uint32_le(unpacked_word);
    }

    // Copy data to device
    float* original_data = nullptr;
    cudaError_t err = cudaMalloc((void**)&original_data, sizeof(float)*float_count);
    if (err != cudaSuccess) { return -1; }
    CallbackScope original_data_cleanup([&]() { cudaFree(original_data); });
    cudaMemcpy(original_data, float_data, sizeof(float)*float_count, cudaMemcpyHostToDevice);

    int block_count = (float_count + BLOCK_FLOAT_COUNT - 1) / BLOCK_FLOAT_COUNT;
    const int block_bytes = BLOCK_PARAM_COUNT*PARAM_SIZE + BLOCK_FLOAT_COUNT*sizeof(float);

    // Create output buffer
    uint8_t* compressed_blocks = nullptr;
    cudaError_t err = cudaMallocManaged((void**)&compressed_blocks, block_count*block_bytes + block_count*sizeof(uint32_t));
    if (err != cudaSuccess) { return -1; }
    CallbackScope original_data_cleanup([&]() { cudaFree(compressed_blocks); });
    uint32_t* block_used_words = reinterpret_cast<uint32_t*>(compressed_blocks + block_count*block_bytes);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    CallbackScope stream_cleanup([&]() { cudaStreamDestroy(stream); });
}

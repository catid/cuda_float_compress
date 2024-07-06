#include "cuszplus_f32.h"

#include <zstd.h>

#include <cuda_runtime.h>
#include <cub/cub.cuh> // CUB from CUDA Toolkit

// FIXME: REMOVE THIS
#include <stdio.h>

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

__device__ inline uint32_t pack_bits(
    const uint32_t* const __restrict__ quant_group,
    uint32_t bit_pos)
{
    uint8_t result = 0;
    uint32_t mask = 1U << bit_pos;

    #pragma unroll
    for (uint32_t j = 0; j < 8; ++j) {
        result |= ((quant_group[j] & mask) != 0) << (7 - j);
    }

    return result;
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

        for (uint32_t j = 0; j < bits; j++) {
            const uint32_t packed_data =
                pack_bits(quant_group + i * QUANT_GROUP_SIZE,      j) |
                pack_bits(quant_group + i * QUANT_GROUP_SIZE + 8,  j) << 8  |
                pack_bits(quant_group + i * QUANT_GROUP_SIZE + 16, j) << 16 |
                pack_bits(quant_group + i * QUANT_GROUP_SIZE + 24, j) << 24;

            *compressed_words++ = packed_data;
        }
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

    // Resize the result vector to the actual compressed size
    Result.resize(output.pos);

    return true;
}


//------------------------------------------------------------------------------
// Decompression Kernel

__global__ void SZplus_decompress_kernel_f32(
    float* const __restrict__ decData,
    const uint8_t* const __restrict__ cmpData,
    volatile uint32_t* const __restrict__ cmpOffset,
    volatile int* const __restrict__ flag,
    const float eb,
    const size_t nbEle)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int block_num = dec_chunk_f32/32;
    const int start_idx = idx * dec_chunk_f32;
    const int start_block_idx = start_idx/32;
    const int rate_ofs = (nbEle+31)/32;

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
                tmp_char0 = cmpData[cmp_byte_ofs++];
                tmp_char1 = cmpData[cmp_byte_ofs++];
                tmp_char2 = cmpData[cmp_byte_ofs++];
                tmp_char3 = cmpData[cmp_byte_ofs++];

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
    
}

#include "cuszplus_f32.h"

#include <zstd.h>

#include <iostream>
using namespace std;


//------------------------------------------------------------------------------
// Constants

static const uint32_t kMagic = 0x00010203;
static const uint32_t kHeaderBytes = 4 + 4 + 4; // see below for format

// Hyper-parameters
#define INTERLEAVE_BITS 2 /* Supports: 1, 2, 4, 8 */
#define ZSTD_COMPRESSION_LEVEL 1 /* Zstd */
#define QUANT_GROUP_SIZE 32 /* Only 32 supported */

// Parallelism
#define BLOCK_SIZE 256 /* Threads per block */
#define THREAD_GROUP_COUNT 4 /* Number of quantization groups per thread */

// Derived constants
#define THREAD_FLOAT_COUNT (THREAD_GROUP_COUNT * QUANT_GROUP_SIZE)
#define BLOCK_FLOAT_COUNT (BLOCK_SIZE * THREAD_FLOAT_COUNT)
#define INTERLEAVE_STRIDE (BLOCK_SIZE * THREAD_GROUP_COUNT)
#define BLOCK_BYTES (BLOCK_FLOAT_COUNT * 4)

/*
    Format:
        kMagic(4 bytes)
        FloatCount(4 bytes)
        Epsilon(4 bytes)
        Zstd-Compressed-Data(X bytes)

    Compression Algorithm:

        The input is padded out to a multiple of BLOCK_FLOAT_COUNT.

        First quantize (int32_t) each float by dividing by epsilon:
            X[i] = Torch.Round( Float[i] / epsilon )

        Within each GPU thread, subtract consecutive floats
        for sets of THREAD_FLOAT_COUNT floats:
            X[i] = X[i] - X[i - 1]

        Zig-zag encoding: (x << 1) ^ (x >> 31)
        This puts the sign bit in the least significant bit, so that
        each quantized value becomes an unsigned integer.
            X[i] = ZigZagEncode( X[i] )

        Interleave bits from groups of QUANT_GROUP_SIZE floats.
        So, every 32 bit word of output corresponds to one vertical column
        of bits from QUANT_GROUP_SIZE quantized floats.

        In a second pass, interleave the words of each block, so that
        e.g. all low bit slices are interleaved together in each block.

    The above algorithm runs on GPU massively in parallel.
    This prepares the data for further compression on CPU using Zstd's
    fastest compression mode.

    The header includes all the information needed to decompress the data.


    Decompression Algorithm:

        We Zstd-decompress the data into a large contiguous buffer that is
        shared with the GPU.

        For each GPU thread:

            De-interleave words for each thread from global memory.
            De-interleave bits from THREAD_FLOAT_COUNT words.

            For THREAD_GROUP_COUNT sets of QUANT_GROUP_SIZE floats:
                X[i] = ZigZagDecode( X[i] ), now a 32-bit signed integer.

                Undo delta coding:
                    X[i] = X[i] + X[i - 1]

                Undo quantization:
                    X[i] = X[i] * epsilon

        The result will be the original set of floating point numbers that can be
        read back to the CPU.


    Discussion:

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

    I tried the exception list approach of https://github.com/lemire/FastPFor
    but found that after Zstd compression, the simpler approach was better.

    I tried the 1-bit interleaving approach of cuSZp, but found that after
    Zstd compression, 2-bit interleaving was faster and sometimes better.
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

#define ZIGZAG_ENCODE(x) (((x) << 1) ^ (x >> 31))
#define ZIGZAG_DECODE(x) ((x) >> 1) ^ -((int32_t)(x) & 1)


//------------------------------------------------------------------------------
// GPU Kernels

__device__ inline void interleave_words_1bit(
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

__device__ inline void interleave_words_2bit(
    const uint32_t* const __restrict__ input,
    uint32_t* const __restrict__ output)
{
    static_assert(QUANT_GROUP_SIZE == 32, "QUANT_GROUP_SIZE must be 32");

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

__device__ inline void interleave_words_4bit(
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

__device__ inline void interleave_words_8bit(
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
// GPU De-interleave Kernels

__device__ inline void deinterleave_words_1bit(
    const uint32_t* const __restrict__ input,
    uint32_t* const __restrict__ output)
{
    #pragma unroll
    for (uint32_t i = 0; i < 32; i++) {
        uint32_t result = 0;
        #pragma unroll
        for (uint32_t j = 0; j < 32; j++) {
            result |= ((input[j] >> i) & 1) << j;
        }
        output[i] = result;
    }
}

__device__ inline void deinterleave_words_2bit(
    const uint32_t* const __restrict__ input,
    uint32_t* const __restrict__ output)
{
    #pragma unroll
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t result_0 = 0, result_1 = 0;
        #pragma unroll
        for (uint32_t j = 0; j < 32; j += 2) {
            result_0 |= ((input[j] >> (i*2)) & 3) << j;
            result_1 |= ((input[j+1] >> (i*2)) & 3) << j;
        }
        output[i] = result_0;
        output[i + 16] = result_1;
    }
}

__device__ inline void deinterleave_words_4bit(
    const uint32_t* const __restrict__ input,
    uint32_t* const __restrict__ output)
{
    #pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t result_0 = 0, result_1 = 0, result_2 = 0, result_3 = 0;
        #pragma unroll
        for (uint32_t j = 0; j < 32; j += 4) {
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
    uint32_t* const __restrict__ output)
{
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        uint32_t result_0 = 0, result_1 = 0, result_2 = 0, result_3 = 0;
        uint32_t result_4 = 0, result_5 = 0, result_6 = 0, result_7 = 0;
        #pragma unroll
        for (uint32_t j = 0; j < 32; j += 8) {
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
// Compression Kernel

__global__ void SZplus_compress(
    const float* __restrict__ original_data,
    int float_count,
    float epsilon,
    uint32_t* __restrict__ compressed_words)
{
    // Local registers for interleaved bits for this thread
    uint32_t thread_words[THREAD_FLOAT_COUNT];

    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    original_data += thread_idx * THREAD_FLOAT_COUNT;
    float_count -= thread_idx * THREAD_FLOAT_COUNT;
    compressed_words += blockIdx.x * BLOCK_FLOAT_COUNT;

    // Quantize, delta, zig-zag encode
    epsilon = 1.0f / epsilon;
    int32_t prev = 0;
    int j = 0;
    for (; j < THREAD_FLOAT_COUNT && j < float_count; j++) {
        const int32_t quant = __float2int_rn(original_data[j] * epsilon);
        const int32_t delta = quant - prev;
        prev = quant;
        thread_words[j] = ZIGZAG_ENCODE(delta);
    }
    for (; j < THREAD_FLOAT_COUNT; j++) {
        thread_words[j] = 0;
    }

    compressed_words += threadIdx.x * THREAD_GROUP_COUNT;

    // Interleave bits for words from this thread
    #pragma unroll
    for (int i = 0; i < THREAD_GROUP_COUNT; i++) {
        uint32_t shuffled_words[QUANT_GROUP_SIZE];

#if INTERLEAVE_BITS == 1
        interleave_words_1bit(
            thread_words + i * QUANT_GROUP_SIZE,
            shuffled_words);
#elif INTERLEAVE_BITS == 2
        interleave_words_2bit(
            thread_words + i * QUANT_GROUP_SIZE,
            shuffled_words);
#elif INTERLEAVE_BITS == 4
        interleave_words_4bit(
            thread_words + i * QUANT_GROUP_SIZE,
            shuffled_words);
#elif INTERLEAVE_BITS == 8
        interleave_words_8bit(
            thread_words + i * QUANT_GROUP_SIZE,
            shuffled_words);
#else
#error "Invalid INTERLEAVE_BITS value. Must be 1, 2, 4, or 8."
#endif

        #pragma unroll
        for (int j = 0; j < QUANT_GROUP_SIZE; j++) {
            compressed_words[j * INTERLEAVE_STRIDE] = shuffled_words[j];
        }
        compressed_words++;
    }
}


//------------------------------------------------------------------------------
// Decompression Kernel

__global__ void SZplus_decompress(
    float* __restrict__ decompressed_floats,
    int float_count,
    const uint32_t* __restrict__ compressed_words,
    float epsilon)
{
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    decompressed_floats += thread_idx * THREAD_FLOAT_COUNT;
    float_count -= thread_idx * THREAD_FLOAT_COUNT;
    compressed_words += BLOCK_FLOAT_COUNT * blockIdx.x;
    compressed_words += threadIdx.x * THREAD_GROUP_COUNT;

    uint32_t shuffled_words[QUANT_GROUP_SIZE];
    uint32_t quant_words[QUANT_GROUP_SIZE];

    int32_t value = 0;

    #pragma unroll
    for (int i = 0; i < THREAD_GROUP_COUNT; i++) {
        #pragma unroll
        for (int j = 0; j < QUANT_GROUP_SIZE; j++) {
            shuffled_words[j] = compressed_words[j * INTERLEAVE_STRIDE];
        }
        compressed_words++;

#if INTERLEAVE_BITS == 1
        deinterleave_words_1bit(shuffled_words, quant_words);
#elif INTERLEAVE_BITS == 2
        deinterleave_words_2bit(shuffled_words, quant_words);
#elif INTERLEAVE_BITS == 4
        deinterleave_words_4bit(shuffled_words, quant_words);
#elif INTERLEAVE_BITS == 8
        deinterleave_words_8bit(shuffled_words, quant_words);
#endif

        for (int j = 0; j < QUANT_GROUP_SIZE; j++) {
            value += ZIGZAG_DECODE(quant_words[j]);
            const float f = value * epsilon;
            if (j < float_count) {
                decompressed_floats[j] = f;
            }
        }

        decompressed_floats += QUANT_GROUP_SIZE;
        float_count -= QUANT_GROUP_SIZE;
    } // next quantization group
}


//------------------------------------------------------------------------------
// Compress API

int GetCompressedBufferSize(int float_count)
{
    const int block_count = (float_count + BLOCK_FLOAT_COUNT - 1) / BLOCK_FLOAT_COUNT;
    return kHeaderBytes + ZSTD_compressBound(block_count * BLOCK_BYTES);
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
    if (compressed_bytes < GetCompressedBufferSize(float_count)) {
        cerr << "ERROR: Compressed buffer is too small. Expected at least " << GetCompressedBufferSize(float_count) <<
            " bytes, but got " << compressed_bytes << " bytes.  Call GetCompressedBufferSize to get the required size." << endl;
        return false;
    }

    // If it's not a device pointer, we need to copy the data to the device
    float* d_original_data = nullptr;
    if (!is_device_ptr) {
        cudaError_t err = cudaMallocAsync((void**)&d_original_data, sizeof(float) * float_count, stream);
        if (err != cudaSuccess) {
            cerr << "cudaMalloc failed: err=" << cudaGetErrorString(err) << " float_count=" << float_count << endl;
            return false;
        }
        cudaMemcpyAsync(d_original_data, float_data, sizeof(float) * float_count, cudaMemcpyHostToDevice, stream);
        float_data = d_original_data;
    }
    CallbackScope original_data_cleanup([&]() { if (d_original_data) cudaFreeAsync(d_original_data, stream); });

    const int block_count = (float_count + BLOCK_FLOAT_COUNT - 1) / BLOCK_FLOAT_COUNT;

    // Create output buffer
    uint32_t* packed_blocks = nullptr;
    cudaError_t err = cudaMallocManaged((void**)&packed_blocks, block_count * BLOCK_BYTES, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        cerr << "cudaMallocAsync failed: err=" << cudaGetErrorString(err) << " block_count=" << block_count << endl;
        return false;
    }
    CallbackScope original_comp_cleanup([&]() { cudaFree(packed_blocks); });

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(block_count);
    SZplus_compress<<<gridSize, blockSize, 0, stream>>>(
        float_data,
        float_count,
        epsilon,
        packed_blocks);

    cudaStreamSynchronize(stream);
    if (cudaSuccess != cudaGetLastError()) {
        cerr << "ERROR: Encountered a CUDA error during FloatCompressor::Compress" << endl;
        return false;
    }

    // Prepare header
    write_uint32_le(compressed_buffer, kMagic);
    write_uint32_le(compressed_buffer + 4, float_count);
    write_float_le(compressed_buffer + 8, epsilon);

    size_t const compressed_size = ZSTD_compress(
        compressed_buffer + kHeaderBytes, compressed_bytes - kHeaderBytes,
        packed_blocks, block_count * BLOCK_BYTES,
        ZSTD_COMPRESSION_LEVEL);
    if (ZSTD_isError(compressed_size)) {
        cerr << "ERROR: ZSTD_compress failed: compressed_size=" << compressed_size << endl;
        return false;
    }

    compressed_bytes = static_cast<int>(kHeaderBytes + compressed_size);
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

    const uint8_t* header_data = static_cast<const uint8_t*>(compressed_data);
    const uint32_t magic = read_uint32_le(header_data);
    const int float_count = read_uint32_le(header_data + 4);

    // Check magic number
    if (magic != kMagic) {
        cerr << "ERROR: Invalid magic number (header parse). Expected 0x" << hex << kMagic << ", but got 0x" << magic << dec << endl;
        return -1;  // Return -1 to indicate an error
    }

    return float_count;
}

bool DecompressFloats(
    cudaStream_t stream,
    const void* compressed_data_,
    int compressed_bytes,
    float* decompressed_floats)
{
    if (compressed_bytes < kHeaderBytes) {
        cerr << "ERROR: Compressed data is too small. Expected at least " << kHeaderBytes << " bytes, but got " << compressed_bytes << " bytes." << endl;
        return false;
    }

    const uint8_t* compressed_data = static_cast<const uint8_t*>(compressed_data_);

    // Read header
    const uint32_t magic = read_uint32_le(compressed_data);
    const int float_count = read_uint32_le(compressed_data + 4);
    const float epsilon = read_float_le(compressed_data + 8);

    if (magic != kMagic) {
        cerr << "ERROR: Invalid magic number. Expected 0x" << hex << kMagic << ", but got 0x" << magic << dec << endl;
        return false;
    }

    const int block_count = (float_count + BLOCK_FLOAT_COUNT - 1) / BLOCK_FLOAT_COUNT;

    // Allocate CUDA managed memory for decompressed blocks
    uint32_t* interleaved_words = nullptr;
    cudaError_t err = cudaMallocManaged((void**)&interleaved_words, block_count * BLOCK_BYTES, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        cerr << "ERROR: Failed to allocate CUDA managed memory for decompressed blocks: " << cudaGetErrorString(err) << endl;
        return false;
    }
    CallbackScope words_cleanup([&]() { cudaFree(interleaved_words); });

    size_t const decompressed_size = ZSTD_decompress(
        interleaved_words, block_count * BLOCK_BYTES,
        compressed_data + kHeaderBytes, compressed_bytes - kHeaderBytes);
    if (ZSTD_isError(decompressed_size)) {
        std::cerr << "ZSTD_decompress error: " << ZSTD_getErrorName(decompressed_size) << std::endl;
        return false;
    }

    if (decompressed_size != block_count * BLOCK_BYTES) {
        std::cerr << "ERROR: ZSTD decompressed output does not match expected size: decompressed_size="
            << decompressed_size << " expected_size=" << block_count * BLOCK_BYTES << std::endl;
        return false;
    }

    // Launch decompression kernel
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(block_count);
    SZplus_decompress<<<gridSize, blockSize, 0, stream>>>(
        decompressed_floats,
        float_count,
        interleaved_words,
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

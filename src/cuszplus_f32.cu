#include "cuszplus_f32.h"

#include <cub/cub.cuh> // CUB from CUDA Toolkit

#define BLOCK_SIZE 256
#define QUANT_GROUP_SIZE 32
#define THREAD_GROUP_COUNT 4
#define THREAD_FLOAT_COUNT (THREAD_GROUP_COUNT * QUANT_GROUP_SIZE)
#define BLOCK_FLOAT_COUNT (BLOCK_SIZE * THREAD_FLOAT_COUNT)
#define BLOCK_PARAM_COUNT (BLOCK_SIZE * THREAD_GROUP_COUNT)
#define PARAM_SIZE (4 + 1 + 1)

// FIXME: REMOVE THIS
#include <stdio.h>

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
    const size_t original_float_count,
    const float epsilon,
    uint8_t* __restrict__ compressed_data)
{
    using BlockScan = cub::BlockScan<uint32_t, BLOCK_SIZE>;
    using BlockAdjacentDifferenceT = cub::BlockAdjacentDifference<int32_t, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    /*
        Block layout:
            MaxIndex(1 byte) x BLOCK_PARAM_COUNT
            Bits(1 byte) x BLOCK_PARAM_COUNT
            HighBits(4 bytes) x BLOCK_PARAM_COUNT
            QuantGroup(4 bytes) x BLOCK_FLOAT_COUNT
    */
    compressed_data += blockIdx.x * (BLOCK_PARAM_COUNT * PARAM_SIZE + BLOCK_FLOAT_COUNT * sizeof(float));

    uint32_t quant_group[THREAD_FLOAT_COUNT];
    uint8_t group_bits[THREAD_GROUP_COUNT];

    uint32_t offset = 0;
    const float inv_epsilon = __frcp_rn(epsilon);
    int32_t prev_quant = 0;
    for (int i = 0; i < THREAD_GROUP_COUNT; i++) {
        uint32_t max_quant = 0, max2_quant = 0;
        uint8_t max_index = 0;

        for (int j = 0; j < QUANT_GROUP_SIZE; j++) {
            int float_index = thread_idx * THREAD_FLOAT_COUNT + j;
            float f = original_data[float_index] * inv_epsilon;

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
        offset += bits * QUANT_GROUP_SIZE / sizeof(uint32_t);

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
    block_scan.InclusiveSum(offset, offset);

    __syncthreads(); // Barrier for smem reuse

    // Get pointer to compressed words
    uint32_t* __restrict__ compressed_words = reinterpret_cast<uint32_t*>(compressed_data + BLOCK_PARAM_COUNT * PARAM_SIZE);

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
// API

int SZplus_compress_hostptr_f32(
    float* oriData,
    uint8_t* cmpBytes,
    size_t nbEle,
    size_t* cmpSize,
    float errorBound)
{
    int block_count = (nbEle + BLOCK_FLOAT_COUNT - 1) / BLOCK_FLOAT_COUNT;

    // Copy data to device
    float* original_data = nullptr;
    cudaError_t err = cudaMalloc((void**)&original_data, sizeof(float)*nbEle);
    if (err != cudaSuccess) { return -1; }
    CallbackScope original_data_cleanup([&]() { cudaFree(original_data); });
    cudaMemcpy(original_data, oriData, sizeof(float)*nbEle, cudaMemcpyHostToDevice);

    err = cudaMalloc((void**)&d_cmpData, sizeof(float)*pad_nbEle);
    if (err != cudaSuccess) { return -1; }

    err = cudaMallocManaged((void**)&d_cmpOffset, sizeof(uint32_t)*cmpOffSize);
    if (err != cudaSuccess) { return -1; }

    cudaMemset(d_cmpOffset, 0, sizeof(uint32_t)*cmpOffSize);
    err = cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    if (err != cudaSuccess) { return -1; }

    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);
    cudaMemset(d_oriData + nbEle, 0, (pad_nbEle - nbEle) * sizeof(float));

    // Initializing CUDA Stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // cuSZp GPU compression.
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(block_count);
    SZplus_compress_kernel_f32<<<gridSize, blockSize, 0, stream>>>(d_oriData, d_cmpData, d_cmpOffset, d_flag, errorBound, nbEle);
    cudaDeviceSynchronize();

    // Obtain compression ratio and move data back to CPU.
    *cmpSize = (size_t)d_cmpOffset[cmpOffSize-1] + (nbEle+31)/32;
    cudaMemcpy(cmpBytes, d_cmpData, *cmpSize, cudaMemcpyDeviceToHost);

    // Free memory that is used.
    cudaFree(d_oriData);
    cudaFree(d_cmpData);
    cudaFree(d_cmpOffset);
    cudaFree(d_flag);
    cudaStreamDestroy(stream);

    return 0;
}

int SZplus_decompress_hostptr_f32(
    float* decData,
    uint8_t* cmpBytes,
    size_t nbEle,
    size_t cmpSize,
    float errorBound)
{
    // Data blocking.
    int bsize = dec_tblock_size_f32;
    int gsize = (nbEle + bsize * dec_chunk_f32 - 1) / (bsize * dec_chunk_f32);
    int cmpOffSize = gsize + 1;
    int pad_nbEle = gsize * bsize * dec_chunk_f32;

    // Initializing global memory for GPU compression.
    float* d_decData = NULL;
    uint8_t* d_cmpData = NULL;
    uint32_t* d_cmpOffset = NULL;
    int* d_flag = NULL;
    cudaError_t err = cudaMalloc((void**)&d_decData, sizeof(float)*pad_nbEle);
    if (err != cudaSuccess) { return -1; }
    //cudaMemset(d_decData, 0, sizeof(float)*pad_nbEle);
    err = cudaMalloc((void**)&d_cmpData, sizeof(float)*pad_nbEle);
    if (err != cudaSuccess) { return -1; }
    cudaMemcpy(d_cmpData, cmpBytes, cmpSize, cudaMemcpyHostToDevice);
    err = cudaMalloc((void**)&d_cmpOffset, sizeof(uint32_t)*cmpOffSize);
    if (err != cudaSuccess) { return -1; }
    cudaMemset(d_cmpOffset, 0, sizeof(uint32_t)*cmpOffSize);
    err = cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    if (err != cudaSuccess) { return -1; }
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Initializing CUDA Stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    printf("gridSize.x: %d\n", gridSize.x);
    printf("blockSize.x: %d\n", blockSize.x);
    SZplus_decompress_kernel_f32<<<gridSize, blockSize, 0, stream>>>(d_decData, d_cmpData, d_cmpOffset, d_flag, errorBound, nbEle);
    cudaDeviceSynchronize();

    // Move data back to CPU.
    cudaMemcpy(decData, d_decData, sizeof(float)*nbEle, cudaMemcpyDeviceToHost);

    // Free memoy that is used.
    cudaFree(d_decData);
    cudaFree(d_cmpData);
    cudaFree(d_cmpOffset);
    cudaFree(d_flag);
    cudaStreamDestroy(stream);

    return 0;
}

int SZplus_compress_deviceptr_f32(
    float* d_oriData,
    uint8_t* d_cmpBytes,
    size_t nbEle,
    size_t* cmpSize,
    float errorBound,
    cudaStream_t stream)
{
    // Data blocking.
    int bsize = cmp_tblock_size_f32;
    int gsize = (nbEle + bsize * cmp_chunk_f32 - 1) / (bsize * cmp_chunk_f32);
    int cmpOffSize = gsize + 1;
    int pad_nbEle = gsize * bsize * cmp_chunk_f32;

    // Initializing global memory for GPU compression.
    uint32_t* d_cmpOffset = NULL;
    int* d_flag = NULL;
    cudaError_t err = cudaMallocManaged((void**)&d_cmpOffset, sizeof(uint32_t)*cmpOffSize);
    if (err != cudaSuccess) { return -1; }
    cudaMemset(d_cmpOffset, 0, sizeof(uint32_t)*cmpOffSize);
    err = cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    if (err != cudaSuccess) { return -1; }
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);
    cudaMemset(d_oriData + nbEle, 0, (pad_nbEle - nbEle) * sizeof(float));

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    SZplus_compress_kernel_f32<<<gridSize, blockSize, 0, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_flag, errorBound, nbEle);
    cudaDeviceSynchronize();

    // Obtain compression ratio and move data back to CPU.  
    *cmpSize = (size_t)d_cmpOffset[cmpOffSize-1] + (nbEle+31)/32;

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_flag);

    return 0;
}

int SZplus_decompress_deviceptr_f32(
    float* d_decData,
    uint8_t* d_cmpBytes,
    size_t nbEle,
    size_t cmpSize,
    float errorBound,
    cudaStream_t stream)
{
    // Data blocking.
    int bsize = dec_tblock_size_f32;
    int gsize = (nbEle + bsize * dec_chunk_f32 - 1) / (bsize * dec_chunk_f32);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    uint32_t* d_cmpOffset = NULL;
    int* d_flag = NULL;
    cudaError_t err = cudaMalloc((void**)&d_cmpOffset, sizeof(uint32_t)*cmpOffSize);
    if (err != cudaSuccess) { return -1; }
    cudaMemset(d_cmpOffset, 0, sizeof(uint32_t)*cmpOffSize);
    err = cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    if (err != cudaSuccess) { return -1; }
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    SZplus_decompress_kernel_f32<<<gridSize, blockSize, 0, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_flag, errorBound, nbEle);
    cudaDeviceSynchronize();

    // Free memoy that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_flag);

    return 0;
}

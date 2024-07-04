#include "cuszplus_f32.h"

// FIXME: REMOVE THIS
#include <stdio.h>

//------------------------------------------------------------------------------
// Tools

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

__device__ inline uint32_t pack_bits(const uint32_t* absQuant, int bit_pos) {
    uint8_t result = 0;
    uint32_t mask = 1U << bit_pos;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        result |= ((absQuant[j] & mask) != 0) << (7 - j);
    }
    return result;
}

// Returns number of bytes written
__device__ inline int pack_block_bits(
    const uint32_t* absQuant,
    uint8_t* cmpData,
    int rate
) {
    for (int i = 0; i < rate; i++) {
        int offset = i * sizeof(uint32_t);
        const uint32_t packed_data = 
            pack_bits(absQuant + offset,      i) |
            pack_bits(absQuant + offset + 8,  i) << 8  |
            pack_bits(absQuant + offset + 16, i) << 16 |
            pack_bits(absQuant + offset + 24, i) << 24;

        // Safe unaligned store
        __stcg(cmpData + offset, packed_data);
    }

    return rate * sizeof(uint32_t);
}

__device__ uint32_t sync_offsets(
    uint32_t thread_ofs,
    const size_t nbEle
)
{
    __shared__ uint32_t base_idx;

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int lane = idx & 31; // Position of thread within its warp
    const int warp = idx >> 5; // Group of 32 threads (CUDA execution unit)
    const int rate_ofs = (nbEle + 31) / 32;

    __syncthreads();

    // thread_ofs = prefix sum(all lower-numbered lanes)
    for (int i = 1; i < 32; i <<= 1) {
        // Read value from lower lane
        uint32_t sum = __shfl_up_sync(0xffffffff/* all threads participate */, thread_ofs, i/* how many lanes to read from */);
        if (lane >= i) {
            thread_ofs += sum;
        }
    }

    __syncthreads();

    if (lane == 31) {
        // Round up to nearest multiple of 8
        cmpOffset[warp + 1] = (thread_ofs + 7) / 8;
        flag[warp + 1] = (warp == 0) ? 2 : 1;
    }

    __syncthreads();

    if (warp > 0 && lane == 0)
    {
        int temp_flag = 1;
        while (temp_flag != 2) {
            temp_flag = flag[warp];
        }

        __threadfence();

        cmpOffset[warp] += cmpOffset[warp - 1];

        if (warp == gridDim.x - 1) {
            cmpOffset[warp + 1] += cmpOffset[warp];
        }

        __threadfence();

        flag[warp + 1] = 2;
    }

    __syncthreads();

    if (lane == 0) {
        base_idx = cmpOffset[warp] + rate_ofs;
    }

    __syncthreads();

    uint32_t cmp_byte_ofs = base_idx;

    if (lane != 0) {
        const uint32_t prev_thread = __shfl_up_sync(0xffffffff, thread_ofs, 1);
        cmp_byte_ofs += prev_thread / 8;
    }

    return cmp_byte_ofs;
}


//------------------------------------------------------------------------------
// Compression Kernel

__global__ void SZplus_compress_kernel_f32(
    const float* const __restrict__ oriData,
    uint8_t* const __restrict__ cmpData,
    volatile uint32_t* const __restrict__ cmpOffset,
    volatile int* const __restrict__ flag,
    const float eb,
    const size_t nbEle)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int block_num = cmp_chunk_f32 / 32;
    const int start_idx = idx * cmp_chunk_f32;
    const int start_block_idx = start_idx / 32;
    const int rate_ofs = (nbEle + 31) / 32;
    const float recipPrecision = __frcp_rn(eb);

    uint32_t absQuant[32];
    uint32_t fixed_rate[block_num];
    uint32_t thread_ofs = 0;

    for(int j = 0; j < block_num; j++)
    {
        int temp_start_idx = start_idx + j * 32;
        int32_t prevQuant = 0;
        uint32_t maxQuant = 0;

        for(int i = 0; i < 32; i++)
        {
            // This is the same quantization used by torch.round()
            const int32_t currQuant = __float2int_rn(oriData[temp_start_idx + i] * recipPrecision);
            const int32_t lorenQuant = currQuant - prevQuant;
            prevQuant = currQuant;

            const uint32_t zigQuant = zigzag_encode(lorenQuant);

            absQuant[i] = zigQuant;
            maxQuant = maxQuant > zigQuant ? maxQuant : zigQuant;
        }

        const uint32_t rate = bit_count(maxQuant);
        fixed_rate[j] = rate;
        thread_ofs += rate != 0 ? rate*32 : 0;

        // FIXME: Why does this happen?
        if (start_block_idx + j < rate_ofs) {
            cmpData[start_block_idx + j] = (uint8_t)rate;
        }
    }

    uint32_t cmp_byte_ofs = sync_offsets(thread_ofs, cmpOffset, flag, nbEle);

    for (int j = 0; j < block_num; j++)
    {
        int chunk_idx_start = j * 32;
        int rate = fixed_rate[j];

        if (rate != 0)
        {
            const int written_bytes = pack_block_bits(
                absQuant + chunk_idx_start,
                cmpData + cmp_byte_ofs,
                rate);

            cmp_byte_ofs += written_bytes;
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
    // Data blocking.
    int bsize = cmp_tblock_size_f32;
    int gsize = (nbEle + bsize * cmp_chunk_f32 - 1) / (bsize * cmp_chunk_f32);
    int cmpOffSize = gsize + 1;
    int pad_nbEle = gsize * bsize * cmp_chunk_f32 * 2;

    // Initializing global memory for GPU compression.
    float* d_oriData = NULL;
    uint8_t* d_cmpData = NULL;
    uint32_t* d_cmpOffset = NULL;
    int* d_flag = NULL;
    cudaError_t err = cudaMalloc((void**)&d_oriData, sizeof(float)*pad_nbEle);
    printf("pad_nbEle: %zu\n", pad_nbEle);
    printf("d_oriData: %p\n", d_oriData);
    printf("err = %d\n", err);
    printf("gsize: %zu\n", gsize);
    printf("bsize: %zu\n", bsize);
    printf("cmp_chunk_f32: %zu\n", cmp_chunk_f32);
    printf("nbEle: %zu\n", nbEle);
    if (err != cudaSuccess) { return -1; }

    cudaMemcpy(d_oriData, oriData, sizeof(float)*nbEle, cudaMemcpyHostToDevice);
    printf("pad_nbEle: %zu\n", pad_nbEle);
    err = cudaMalloc((void**)&d_cmpData, sizeof(float)*pad_nbEle);
    if (err != cudaSuccess) { return -1; }

    printf("cmpOffSize: %zu\n", cmpOffSize);
    err = cudaMallocManaged((void**)&d_cmpOffset, sizeof(uint32_t)*cmpOffSize);
    if (err != cudaSuccess) { return -1; }

    cudaMemset(d_cmpOffset, 0, sizeof(uint32_t)*cmpOffSize);
    err = cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    printf("cmpOffSize: %zu\n", cmpOffSize);
    if (err != cudaSuccess) { return -1; }

    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);
    cudaMemset(d_oriData + nbEle, 0, (pad_nbEle - nbEle) * sizeof(float));

    // Initializing CUDA Stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    const int sharedMemSize = bsize * gsize * sizeof(uint32_t);
    SZplus_compress_kernel_f32<<<gridSize, blockSize, sharedMemSize, stream>>>(d_oriData, d_cmpData, d_cmpOffset, d_flag, errorBound, nbEle);
    cudaDeviceSynchronize();

    // Obtain compression ratio and move data back to CPU.
    *cmpSize = (size_t)d_cmpOffset[cmpOffSize-1] + (nbEle+31)/32;
    cudaMemcpy(cmpBytes, d_cmpData, *cmpSize, cudaMemcpyDeviceToHost);

    printf("sizeof(float)*pad_nbEle = %zu\n", sizeof(float)*pad_nbEle);
    printf("*cmpSize = %d\n", (int)*cmpSize);

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

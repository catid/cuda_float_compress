#include "cuszplus_f32.h"

#include <stdio.h>

__device__ inline int quantization_f32(float data, float recipPrecision)
{
    float dataRecip = data*recipPrecision;
    int s = dataRecip>=-0.5f?0:1;
    return (int)(dataRecip+0.5f) - s;
}

__device__ inline int get_bit_num(unsigned int x)
{
    return (sizeof(unsigned int)*8) - __clz(x);
}

__global__ void SZplus_compress_kernel_f32(
    const float* const __restrict__ oriData,
    unsigned char* const __restrict__ cmpData,
    volatile unsigned int* const __restrict__ cmpOffset,
    volatile int* const __restrict__ flag,
    const float eb,
    const size_t nbEle)
{
    __shared__ unsigned int base_idx;

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int lane = idx & 31;
    const int warp = idx >> 5;
    const int block_num = cmp_chunk_f32/32;
    const int start_idx = idx * cmp_chunk_f32;
    const int start_block_idx = start_idx/32;
    const int rate_ofs = (nbEle+31)/32;
    const float recipPrecision = 0.5f/eb;

    int temp_start_idx, temp_end_idx;
    int quant_chunk_idx;
    int block_idx;
    int currQuant, lorenQuant, prevQuant, maxQuant;
    int absQuant[cmp_chunk_f32];
    unsigned int sign_flag[block_num];
    int sign_ofs;
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0;

    for(int j=0; j<block_num; j++)
    {
        sign_flag[j] = 0;
        temp_start_idx = start_idx + j*32;
        temp_end_idx = temp_start_idx + 32;
        block_idx = start_block_idx+j;
        prevQuant = 0;
        maxQuant = 0;

        for(int i=temp_start_idx; i<temp_end_idx; i++)
        {
            quant_chunk_idx = i%cmp_chunk_f32;
            currQuant = quantization_f32(oriData[i], recipPrecision);
            lorenQuant = currQuant - prevQuant;
            prevQuant = currQuant;
            sign_ofs = i % 32;
            sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
            absQuant[quant_chunk_idx] = abs(lorenQuant);
            maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant : absQuant[quant_chunk_idx];
        }

        fixed_rate[j] = get_bit_num(maxQuant);
        thread_ofs += (fixed_rate[j]) ? (32+fixed_rate[j]*32) : 0;
        if(block_idx<rate_ofs) cmpData[block_idx] = (unsigned char)fixed_rate[j];
    }
    __syncthreads();

    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    if(lane==31) 
    {
        cmpOffset[warp+1] = (thread_ofs+7)/8;
        __threadfence();
        if(warp==0)
        {
            flag[1] = 2;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();
        }
    }
    __syncthreads();

    if(warp>0)
    {
        if(!lane)
        {
            int temp_flag = 1;
            while(temp_flag!=2) temp_flag = flag[warp];
            __threadfence();
            cmpOffset[warp] += cmpOffset[warp-1];
            if(warp==gridDim.x-1) cmpOffset[warp+1] += cmpOffset[warp];
            __threadfence();
            flag[warp+1] = 2;
        }
        
    }
    __syncthreads();

    if(!lane) base_idx = cmpOffset[warp] + rate_ofs;
    __syncthreads();

    unsigned int prev_thread = __shfl_up_sync(0xffffffff, thread_ofs, 1);
    unsigned int cmp_byte_ofs;
    if(!lane) cmp_byte_ofs = base_idx;
    else cmp_byte_ofs = base_idx + prev_thread / 8;
    
    for(int j=0; j<block_num; j++)  
    {
        int chunk_idx_start = j*32;
        int rate = fixed_rate[j];

        if(rate != 0)
        {
            cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 24);
            cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 16);
            cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 8);
            cmpData[cmp_byte_ofs++] = 0xff & sign_flag[j];

            unsigned char tmp_char0, tmp_char1, tmp_char2, tmp_char3;
            int mask = 1;
            for(int i=0; i<rate; i++)
            {
                tmp_char0 = 0;
                tmp_char1 = 0;
                tmp_char2 = 0;
                tmp_char3 = 0;

                tmp_char0 = (((absQuant[chunk_idx_start+0] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+1] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+2] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+3] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+4] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+5] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+6] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+7] & mask) >> i) << 0);

                tmp_char1 = (((absQuant[chunk_idx_start+8] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+9] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+10] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+11] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+12] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+13] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+14] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+15] & mask) >> i) << 0);

                tmp_char2 = (((absQuant[chunk_idx_start+16] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+17] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+18] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+19] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+20] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+21] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+22] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+23] & mask) >> i) << 0);
                
                tmp_char3 = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);

                // Move data to global memory.
                cmpData[cmp_byte_ofs++] = tmp_char0;
                cmpData[cmp_byte_ofs++] = tmp_char1;
                cmpData[cmp_byte_ofs++] = tmp_char2;
                cmpData[cmp_byte_ofs++] = tmp_char3;
                mask <<= 1;
            }
        }
    }
}

__global__ void SZplus_decompress_kernel_f32(
    float* const __restrict__ decData,
    const unsigned char* const __restrict__ cmpData,
    volatile unsigned int* const __restrict__ cmpOffset,
    volatile int* const __restrict__ flag,
    const float eb,
    const size_t nbEle)
{
    __shared__ unsigned int base_idx;

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int lane = idx & 31;
    const int warp = idx >> 5;
    const int block_num = dec_chunk_f32/32;
    const int start_idx = idx * dec_chunk_f32;
    const int start_block_idx = start_idx/32;
    const int rate_ofs = (nbEle+31)/32;

    int temp_start_idx;
    int block_idx;
    int absQuant[32];
    int currQuant, lorenQuant, prevQuant;
    int sign_ofs;
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0;

    for(int j=0; j<block_num; j++)
    {
        block_idx = start_block_idx + j;
        if(block_idx<rate_ofs) 
        {
            fixed_rate[j] = (int)cmpData[block_idx];
            thread_ofs += (fixed_rate[j]) ? (32+fixed_rate[j]*32) : 0;
        }
    }
    __syncthreads();

    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    if(lane==31) 
    {
        cmpOffset[warp+1] = (thread_ofs+7)/8;
        __threadfence();
        if(warp==0)
        {
            flag[1] = 2;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();
        }
    }
    __syncthreads();

    if(warp>0)
    {
        if(!lane)
        {
            int temp_flag = 1;
            while(temp_flag!=2) temp_flag = flag[warp];
            __threadfence();
            cmpOffset[warp] += cmpOffset[warp-1];
            __threadfence();
            flag[warp+1] = 2;
        }
    }
    __syncthreads();

    if(!lane) base_idx = cmpOffset[warp] + rate_ofs;
    __syncthreads();

    unsigned int prev_thread = __shfl_up_sync(0xffffffff, thread_ofs, 1);
    unsigned int cmp_byte_ofs;
    if(!lane) cmp_byte_ofs = base_idx;
    else cmp_byte_ofs = base_idx + prev_thread / 8;

    for(int j=0; j<block_num; j++)
    {
        temp_start_idx = start_idx + j*32;
        unsigned int sign_flag = 0;

        if(fixed_rate[j])
        {
            sign_flag = (0xff000000 & (cmpData[cmp_byte_ofs++] << 24)) |
                        (0x00ff0000 & (cmpData[cmp_byte_ofs++] << 16)) |
                        (0x0000ff00 & (cmpData[cmp_byte_ofs++] << 8))  |
                        (0x000000ff & cmpData[cmp_byte_ofs++]);
            
            unsigned char tmp_char0, tmp_char1, tmp_char2, tmp_char3;
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
            prevQuant = 0;
            for(int i=0; i<32; i++)
            {
                sign_ofs = i % 32;
                if(sign_flag & (1 << (31 - sign_ofs)))
                    lorenQuant = absQuant[i] * -1;
                else
                    lorenQuant = absQuant[i];
                currQuant = (lorenQuant + prevQuant) * 2;
                decData[temp_start_idx+i] = currQuant * eb;
                prevQuant = currQuant;
            }
        } else {
            for(int i=0; i<32; i++)
            {
                decData[temp_start_idx+i] = 0.f;
            }
        }
    }

    printf("cmp_byte_ofs = %d\n", (int)cmp_byte_ofs);
}

int SZplus_compress_hostptr_f32(
    float* oriData,
    unsigned char* cmpBytes,
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
    unsigned char* d_cmpData = NULL;
    unsigned int* d_cmpOffset = NULL;
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
    err = cudaMallocManaged((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    if (err != cudaSuccess) { return -1; }

    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
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
    SZplus_compress_kernel_f32<<<gridSize, blockSize, 0, stream>>>(d_oriData, d_cmpData, d_cmpOffset, d_flag, errorBound, nbEle);
    cudaDeviceSynchronize();

    // Obtain compression ratio and move data back to CPU.  
    printf("nbEle % 32 = %d\n", (int)(nbEle % 32));
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
    unsigned char* cmpBytes,
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
    unsigned char* d_cmpData = NULL;
    unsigned int* d_cmpOffset = NULL;
    int* d_flag = NULL;
    cudaError_t err = cudaMalloc((void**)&d_decData, sizeof(float)*pad_nbEle);
    if (err != cudaSuccess) { return -1; }
    //cudaMemset(d_decData, 0, sizeof(float)*pad_nbEle);
    err = cudaMalloc((void**)&d_cmpData, sizeof(float)*pad_nbEle);
    if (err != cudaSuccess) { return -1; }
    cudaMemcpy(d_cmpData, cmpBytes, cmpSize, cudaMemcpyHostToDevice);
    err = cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    if (err != cudaSuccess) { return -1; }
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
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
    unsigned char* d_cmpBytes,
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
    unsigned int* d_cmpOffset = NULL;
    int* d_flag = NULL;
    cudaError_t err = cudaMallocManaged((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    if (err != cudaSuccess) { return -1; }
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
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
    unsigned char* d_cmpBytes,
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
    unsigned int* d_cmpOffset = NULL;
    int* d_flag = NULL;
    cudaError_t err = cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    if (err != cudaSuccess) { return -1; }
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
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

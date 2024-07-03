#pragma once

#include <cuda_runtime.h>

#include <cstdint>

// Returns 0 if successful, otherwise returns error code.
int SZplus_compress_hostptr_f32(
    float* oriData,
    uint8_t* cmpBytes,
    size_t nbEle,
    size_t* cmpSize,
    float errorBound);
int SZplus_decompress_hostptr_f32(
    float* decData,
    uint8_t* cmpBytes,
    size_t nbEle,
    size_t cmpSize,
    float errorBound);
int SZplus_compress_deviceptr_f32(
    float* d_oriData,
    uint8_t* d_cmpBytes,
    size_t nbEle,
    size_t* cmpSize,
    float errorBound,
    cudaStream_t stream = 0);
int SZplus_decompress_deviceptr_f32(
    float* d_decData,
    uint8_t* d_cmpBytes,
    size_t nbEle,
    size_t cmpSize,
    float errorBound,
    cudaStream_t stream = 0);

static const int cmp_tblock_size_f32 = 32;
static const int dec_tblock_size_f32 = 32;
static const int cmp_chunk_f32 = 8192;
static const int dec_chunk_f32 = 8192;

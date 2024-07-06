#pragma once

#include <cuda_runtime.h>

#include <cstdint>

struct FloatCompressor
{
    
};

// Returns 0 if successful, otherwise returns error code.
int SZplus_compress_hostptr_f32(
    float* original_data,
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

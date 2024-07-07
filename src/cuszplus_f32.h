#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>


//------------------------------------------------------------------------------
// Compress API

int GetCompressedBufferSize(int float_count);

bool CompressFloats(
    cudaStream_t stream,
    const float* float_data,
    bool is_device_ptr,
    int float_count,
    uint8_t* compressed_buffer,
    int& compressed_bytes,
    float epsilon = 0.0001f);


//------------------------------------------------------------------------------
// Decompress API

int GetDecompressedFloatCount(
    const void* compressed_data,
    int compressed_bytes);

bool DecompressFloats(
    cudaStream_t stream,
    const void* compressed_data,
    int compressed_bytes,
    float* decompressed_floats);

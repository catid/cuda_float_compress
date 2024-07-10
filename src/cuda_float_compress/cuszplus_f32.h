#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>


//------------------------------------------------------------------------------
// Compress API

// Returns the size of the compressed_buffer required to hold the given number of floats.
int GetCompressedBufferSize(int float_count);

/*
    Compresses the given floats into the given CPU compressed_buffer.
    Precision of the floats are determined by the epsilon parameter.
    Returns true on success, false on failure.

    The compressed_buffer must be at least GetCompressedBufferSize(float_count) bytes in size.
    The compressed_bytes output parameter will be set to the actual size of the compressed data,
    and it must be initialized to GetCompressedBufferSize(float_count) or larger.

    Runs CUDA kernels on the given stream (set to nullptr to use the default stream).
    Pass is_device_ptr=true if the float_data pointer is a CUDA device pointer.

    The compressed_buffer should be on CPU not a CUDA device pointer.
*/
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

// Returns the number of floats in the given compressed data by reading header.
int GetDecompressedFloatCount(
    const void* compressed_data,
    int compressed_bytes);

/*
    Decompresses the given compressed data into the given CPU decompressed_floats.
    Returns true on success, false on failure.

    The decompressed_floats must be at least GetDecompressedFloatCount(compressed_data, compressed_bytes) floats in size.
    The decompressed_floats must be allocated as a CUDA device pointer.

    Runs CUDA kernels on the given stream (set to nullptr to use the default stream).
*/
bool DecompressFloats(
    cudaStream_t stream,
    const void* compressed_data,
    int compressed_bytes,
    float* decompressed_floats);

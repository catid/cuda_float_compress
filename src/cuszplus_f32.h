#pragma once

#include <cstdint>
#include <vector>


//------------------------------------------------------------------------------
// FloatCompressor

struct FloatCompressor {
    bool Compress(
        const float* float_data,
        int float_count,
        float epsilon = 0.0001f);

    // Result of Compress()
    std::vector<uint8_t> Result;

    int GetCompressedBytes() const {
        return static_cast<int>( Result.size() );
    }
};


//------------------------------------------------------------------------------
// FloatDecompressor

struct FloatDecompressor {
    bool Decompress(
        const void* compressed_data,
        int compressed_bytes);

    // Result of Decompress()
    std::vector<float> Result;

    int GetFloatCount() const {
        return static_cast<int>( Result.size() );
    }
};

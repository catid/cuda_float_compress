#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <random>

#include <zstd.h>


//------------------------------------------------------------------------------
// Constants

#define ZSTD_COMPRESSION_LEVEL 1
#define QUANT_GROUP_SIZE 32
#define INTERLEAVE_BITS 2

// This is simulating how in the CUDA kernel each thread works on 32 * 4 floats
#define DELTA_RESET_INTERVAL_GROUPS 32 * 4

// This interleaves all quantization groups of 32 together in a second pass.
#define ENABLE_SECOND_PASS

#ifdef ENABLE_SECOND_PASS
    // Simulates only interleaving within a block of 256 threads
    #define INTERLEAVE_ONLY_LOCAL 256 * DELTA_RESET_INTERVAL_GROUPS
#endif

#ifndef ENABLE_SECOND_PASS
    // This alternative approach writes only non-zero bit-slices.
    // This is incompatible with the second pass.
    #define ENABLE_VARLEN_BITS
#endif

#ifdef ENABLE_VARLEN_BITS
    // This one sets aside the largest value in each quantization group
    // in an exception list and writes their high bits separately.
    #define ENABLE_EXCEPTION_LIST
    // This option enables interleaving the exception list bits.
    #define ENABLE_INTERLEAVE_HIGH_BITS
#endif

/*
    Algorithm:

        For each set of 32 words:
            Subtract neighboring words
            Zigzag encode

    Results:

    Do not interleave groups, do not use varlen bits, do not use high bits:

    1-bit interleave:
*/


//------------------------------------------------------------------------------
// Tools

// Round up to the next power of 2
#define ROUND_UP_POW2(x, pow2) \
    (((x) + ((pow2) - 1)) & ~((pow2) - 1))

// Zigzag encoding
inline uint32_t zigzag_encode(int32_t n) {
    return (n << 1) ^ (n >> 31);
}

// Bit count
inline uint32_t bit_count(uint32_t x) {
    return 32 - __builtin_clz(x | 1);
}


//------------------------------------------------------------------------------
// Encode/Decode Functions

void compress_int32(
    const std::vector<int32_t>& input,
    std::vector<uint8_t>& bit_counts,
    std::vector<uint8_t>& high_index,
    std::vector<uint32_t>& high_bits,
    std::vector<uint32_t>& encoded_data)
{
    size_t input_size = input.size();
    size_t group_count = (input_size + QUANT_GROUP_SIZE - 1) / QUANT_GROUP_SIZE;
    
#ifdef ENABLE_VARLEN_BITS
    bit_counts.resize(group_count);
#endif
#ifdef ENABLE_EXCEPTION_LIST
    high_index.resize(group_count);
    high_bits.resize(group_count);
#endif
    encoded_data.resize(input_size);

    uint32_t prev = 0;

    // For each group: Subtract neighbors, zigzag encode, set aside one exception, and find max bit count.
    for (size_t group = 0; group < group_count; ++group) {
        if (group % DELTA_RESET_INTERVAL_GROUPS == 0) {
            prev = 0;
        }
        size_t start = group * QUANT_GROUP_SIZE;
        size_t end = std::min(start + QUANT_GROUP_SIZE, input_size);

#ifdef ENABLE_EXCEPTION_LIST
        uint32_t max_data = 0, max2_data = 0;
        int max_index = 0;
#else
        uint32_t max_data = 0;
#endif
        for (size_t i = start; i < end; ++i) {
            int32_t diff = input[i] - prev;
            uint32_t data = zigzag_encode(diff);
            encoded_data[i] = data;
            prev = input[i];

#ifdef ENABLE_EXCEPTION_LIST
            if (data > max_data) {
                max2_data = max_data;
                max_data = data;
                max_index = (int)(i - start);
            } else if (data > max2_data) {
                max2_data = data;
            }
#else
            if (data > max_data) {
                max_data = data;
            }
#endif
        }

#ifdef ENABLE_EXCEPTION_LIST
        const int32_t max_bits = ROUND_UP_POW2(bit_count(max2_data), INTERLEAVE_BITS);
#else
        const int32_t max_bits = ROUND_UP_POW2(bit_count(max_data), INTERLEAVE_BITS);
#endif

#ifdef ENABLE_VARLEN_BITS
        bit_counts[group] = static_cast<uint8_t>(max_bits);
#endif
#ifdef ENABLE_EXCEPTION_LIST
        high_bits[group] = max_data >> max_bits;
        high_index[group] = static_cast<uint8_t>(max_index);
#endif
    }
}

void encode_bit_counts(std::vector<uint8_t>& bit_counts)
{
    int32_t prev_bits = 0;

    for (size_t group = 0; group < bit_counts.size(); ++group) {
        const int32_t max_bits = bit_counts[group];
        const int32_t delta_bits = max_bits - prev_bits;
        uint32_t zig_bits = zigzag_encode(delta_bits);
        prev_bits = max_bits;

        bit_counts[group] = static_cast<uint8_t>(zig_bits);
    }
}

void decode_bit_counts(std::vector<uint8_t>& bit_counts)
{
    int32_t max_bits = 0;

    for (size_t group = 0; group < bit_counts.size(); ++group) {
        uint32_t zig_bits = bit_counts[group];
        int32_t delta_bits = (zig_bits >> 1) ^ (-(zig_bits & 1));
        max_bits += delta_bits;

        bit_counts[group] = static_cast<uint8_t>(max_bits);
    }
}

void decompress_int32(
    size_t group_count,
    const std::vector<uint8_t>& bit_counts,
    const std::vector<uint8_t>& high_index,
    const std::vector<uint32_t>& high_bits,
    const std::vector<uint32_t>& encoded_data,
    std::vector<int32_t>& output)
{
    size_t input_size = encoded_data.size();

    output.resize(input_size);

    uint32_t prev = 0;

    for (size_t group = 0; group < group_count; ++group) {
        if (group % DELTA_RESET_INTERVAL_GROUPS == 0) {
            prev = 0;
        }
#ifdef ENABLE_EXCEPTION_LIST
        const uint32_t max_bits = bit_counts[group];
#endif

        size_t start = group * QUANT_GROUP_SIZE;
        size_t end = std::min(start + QUANT_GROUP_SIZE, input_size);

        for (size_t i = start; i < end; ++i) {
            uint32_t encoded = encoded_data[i];
#ifdef ENABLE_EXCEPTION_LIST
            if (i - start == high_index[group]) {
                encoded |= high_bits[group] << max_bits;
            }
#endif
            int32_t decoded = (encoded >> 1) ^ (-(encoded & 1));  // Zigzag decode
            output[i] = decoded + prev;
            prev = output[i];
        }
    }
}


//------------------------------------------------------------------------------
// CPU Version

// Helper function to pack bits from 32 input words into a single output word
uint32_t cpu_pack_bits(const uint32_t* input, uint32_t bit_position) {
    uint32_t result = 0;
    uint32_t mask = 1u << bit_position;
    
    for (int i = 0; i < 32; ++i) {
        // Extract the bit at bit_position from each input word
        uint32_t bit = (input[i] & mask) ? 1 : 0;
        
        // Place this bit in the correct position in the result
        result |= (bit << i);
    }
    
    return result;
}

// Main interleave function
void cpu_interleave_1bit(const uint32_t* input, uint32_t* output, int block_count, int bits) {
    for (int block = 0; block < block_count; ++block) {
        // Pointer to the start of the current input block
        const uint32_t* block_input = input + (block * 32);
        
        // Pointer to the start of the current output block
        uint32_t* block_output = output + (block * bits);
        
        // For each bit position in the 32-bit words
        for (int bit_pos = 0; bit_pos < bits; ++bit_pos) {
            // Pack the bits at this position from all 32 input words in the current block
            block_output[bit_pos] = cpu_pack_bits(block_input, bit_pos);
        }
    }
}


//------------------------------------------------------------------------------
// CPU Version (2 bits at a time)

// Helper function to pack 2 bits from 16 input words into a single output word
uint32_t cpu_pack_2bits(const uint32_t* input, uint32_t shift) {
    uint32_t result = 0;
    uint32_t mask = 0x3 << shift;
    
    for (int i = 0; i < 16; ++i) {
        // Extract 2 bits at shift position from each input word
        uint32_t bits = (input[i] & mask) >> shift;
        
        // Place these 2 bits in the correct position in the result
        result |= (bits << (i * 2));
    }
    
    return result;
}

// Main interleave function
void cpu_interleave_2bit(const uint32_t* input, uint32_t* output, int block_count, int bits) {
    for (int block = 0; block < block_count; ++block) {
        // Pointer to the start of the current input block
        const uint32_t* block_input = input + (block * 32);

        // Pointer to the start of the current output block
        uint32_t* block_output = output + (block * bits);

        // For each 2-bit position in the 32-bit words
        for (int shift = 0; shift < bits; shift += 2) {
            // Pack 2 bits at this position from all 32 input words in the current block
            block_output[shift] = cpu_pack_2bits(block_input, shift);
            block_output[shift + 1] = cpu_pack_2bits(block_input + 16, shift);
        }
    }
}


//------------------------------------------------------------------------------
// CPU Version (4 bits at a time)

// Helper function to pack 4 bits from 8 input words into a single output word
uint32_t cpu_pack_4bits(const uint32_t* input, uint32_t shift) {
    uint32_t result = 0;
    uint32_t mask = 0xF << shift;
    
    for (int i = 0; i < 8; ++i) {
        // Extract 4 bits at shift position from each input word
        uint32_t bits = (input[i] & mask) >> shift;
        
        // Place these 4 bits in the correct position in the result
        result |= (bits << (i * 4));
    }
    
    return result;
}

// Main interleave function
void cpu_interleave_4bit(const uint32_t* input, uint32_t* output, int block_count, int bits) {
    for (int block = 0; block < block_count; ++block) {
        // Pointer to the start of the current input block
        const uint32_t* block_input = input + (block * 32);
        
        // Pointer to the start of the current output block
        uint32_t* block_output = output + (block * bits);
        
        // For each 4-bit position in the 32-bit words
        for (int shift = 0; shift < bits; shift += 4) {
            // Pack 4 bits at this position from all 32 input words in the current block
            block_output[shift] = cpu_pack_4bits(block_input, shift);
            block_output[shift + 1] = cpu_pack_4bits(block_input + 8, shift);
            block_output[shift + 2] = cpu_pack_4bits(block_input + 16, shift);
            block_output[shift + 3] = cpu_pack_4bits(block_input + 24, shift);
        }
    }
}


//------------------------------------------------------------------------------
// CPU Version (8 bits at a time)

// Helper function to pack 8 bits from 4 input words into a single output word
uint32_t cpu_pack_8bits(const uint32_t* input, uint32_t shift) {
    uint32_t result = 0;
    uint32_t mask = 0xFF << shift;
    
    for (int i = 0; i < 4; ++i) {
        // Extract 8 bits at shift position from each input word
        uint32_t bits = (input[i] & mask) >> shift;
        
        // Place these 8 bits in the correct position in the result
        result |= (bits << (i * 8));
    }
    
    return result;
}

// Main interleave function
void cpu_interleave_8bit(const uint32_t* input, uint32_t* output, int block_count, int bits) {
    for (int block = 0; block < block_count; ++block) {
        // Pointer to the start of the current input block
        const uint32_t* block_input = input + (block * 32);

        // Pointer to the start of the current output block
        uint32_t* block_output = output + (block * bits);

        // For each 8-bit position in the 32-bit words
        for (int shift = 0; shift < bits; shift += 8) {
            // Pack 8 bits at this position from all 32 input words in the current block
            block_output[shift] = cpu_pack_8bits(block_input, shift);
            block_output[shift + 1] = cpu_pack_8bits(block_input + 4, shift);
            block_output[shift + 2] = cpu_pack_8bits(block_input + 8, shift);
            block_output[shift + 3] = cpu_pack_8bits(block_input + 12, shift);
            block_output[shift + 4] = cpu_pack_8bits(block_input + 16, shift);
            block_output[shift + 5] = cpu_pack_8bits(block_input + 20, shift);
            block_output[shift + 6] = cpu_pack_8bits(block_input + 24, shift);
            block_output[shift + 7] = cpu_pack_8bits(block_input + 28, shift);
        }
    }
}


//------------------------------------------------------------------------------
// CPU De-interleave Functions

void cpu_deinterleave_1bit(const uint32_t* input, uint32_t* output, int block_count, int low_bits) {
    for (int block = 0; block < block_count; ++block) {
        const uint32_t* block_input = input + (block * low_bits);
        uint32_t* block_output = output + (block * 32);
        
        for (int i = 0; i < 32; ++i) {
            uint32_t result = 0;
            for (int j = 0; j < low_bits; ++j) {
                result |= ((block_input[j] >> i) & 1) << j;
            }
            block_output[i] = result;
        }
    }
}

void cpu_deinterleave_2bit(const uint32_t* input, uint32_t* output, int block_count, int low_bits) {
    for (int block = 0; block < block_count; ++block) {
        const uint32_t* block_input = input + (block * low_bits);
        uint32_t* block_output = output + (block * 32);
        
        for (int i = 0; i < 16; ++i) {
            uint32_t result_0 = 0, result_1 = 0;
            for (int j = 0; j < low_bits; j += 2) {
                result_0 |= ((block_input[j] >> (i*2)) & 3) << j;
                result_1 |= ((block_input[j+1] >> (i*2)) & 3) << j;
            }
            block_output[i] = result_0;
            block_output[i + 16] = result_1;
        }
    }
}

void cpu_deinterleave_4bit(const uint32_t* input, uint32_t* output, int block_count, int low_bits) {
    for (int block = 0; block < block_count; ++block) {
        const uint32_t* block_input = input + (block * low_bits);
        uint32_t* block_output = output + (block * 32);
        
        for (int i = 0; i < 8; ++i) {
            uint32_t result_0 = 0, result_1 = 0, result_2 = 0, result_3 = 0;
            for (int j = 0; j < low_bits; j += 4) {
                result_0 |= ((block_input[j] >> (i*4)) & 0xF) << j;
                result_1 |= ((block_input[j+1] >> (i*4)) & 0xF) << j;
                result_2 |= ((block_input[j+2] >> (i*4)) & 0xF) << j;
                result_3 |= ((block_input[j+3] >> (i*4)) & 0xF) << j;
            }
            block_output[i] = result_0;
            block_output[i + 8] = result_1;
            block_output[i + 16] = result_2;
            block_output[i + 24] = result_3;
        }
    }
}

void cpu_deinterleave_8bit(const uint32_t* input, uint32_t* output, int block_count, int low_bits) {
    for (int block = 0; block < block_count; ++block) {
        const uint32_t* block_input = input + (block * low_bits);
        uint32_t* block_output = output + (block * 32);

        for (int i = 0; i < 4; ++i) {
            uint32_t result_0 = 0, result_1 = 0, result_2 = 0, result_3 = 0;
            uint32_t result_4 = 0, result_5 = 0, result_6 = 0, result_7 = 0;
            for (int j = 0; j < low_bits; j += 8) {
                result_0 |= ((block_input[j] >> (i*8)) & 0xFF) << j;
                result_1 |= ((block_input[j+1] >> (i*8)) & 0xFF) << j;
                result_2 |= ((block_input[j+2] >> (i*8)) & 0xFF) << j;
                result_3 |= ((block_input[j+3] >> (i*8)) & 0xFF) << j;
                result_4 |= ((block_input[j+4] >> (i*8)) & 0xFF) << j;
                result_5 |= ((block_input[j+5] >> (i*8)) & 0xFF) << j;
                result_6 |= ((block_input[j+6] >> (i*8)) & 0xFF) << j;
                result_7 |= ((block_input[j+7] >> (i*8)) & 0xFF) << j;
            }
            block_output[i] = result_0;
            block_output[i + 4] = result_1;
            block_output[i + 8] = result_2;
            block_output[i + 12] = result_3;
            block_output[i + 16] = result_4;
            block_output[i + 20] = result_5;
            block_output[i + 24] = result_6;
            block_output[i + 28] = result_7;
        }
    }
}


//------------------------------------------------------------------------------
// Bit Selection Interposer

void cpu_interleave_bits(const uint32_t* input, uint32_t* output, int block_count, int bits) {
#if INTERLEAVE_BITS == 1
    cpu_interleave_1bit(input, output, block_count, bits);
#elif INTERLEAVE_BITS == 2
    cpu_interleave_2bit(input, output, block_count, bits);
#elif INTERLEAVE_BITS == 4
    cpu_interleave_4bit(input, output, block_count, bits);
#elif INTERLEAVE_BITS == 8
    cpu_interleave_8bit(input, output, block_count, bits);
#endif
}

void cpu_deinterleave_bits(const uint32_t* input, uint32_t* output, int block_count, int bits) {
#if INTERLEAVE_BITS == 1
    cpu_deinterleave_1bit(input, output, block_count, bits);
#elif INTERLEAVE_BITS == 2
    cpu_deinterleave_2bit(input, output, block_count, bits);
#elif INTERLEAVE_BITS == 4
    cpu_deinterleave_4bit(input, output, block_count, bits);
#elif INTERLEAVE_BITS == 8
    cpu_deinterleave_8bit(input, output, block_count, bits);
#endif
}


//------------------------------------------------------------------------------
// Grouped Interleave Varlen Functions

void interleave_to_32(const std::vector<uint32_t>& input, const std::vector<uint8_t>& bit_counts, std::vector<uint32_t>& output) {
    size_t input_size = input.size();
    size_t full_blocks = input_size / 32;
    size_t remaining = input_size % 32;
    
    size_t output_size = 0;
    for (size_t i = 0; i < full_blocks; ++i) {
        output_size += bit_counts[i];
    }
    if (remaining > 0) {
        output_size += bit_counts[full_blocks];
    }

    output.resize(output_size, 0);  // Initialize with zeros
    
    int offset = 0;
    for (size_t i = 0; i < full_blocks; ++i) {
        const uint8_t bits = bit_counts[i];
        cpu_interleave_bits(input.data() + i * 32, output.data() + offset, 1, bits);
        offset += bits;
    }
    
    // Process remaining elements
    if (remaining > 0) {
        const uint8_t bits = bit_counts[full_blocks];
        std::vector<uint32_t> padded_input(32, 0);
        std::memcpy(padded_input.data(), input.data() + full_blocks * 32, remaining * sizeof(uint32_t));
        cpu_interleave_bits(padded_input.data(), output.data() + offset, 1, bits);
    }
}

void deinterleave_from_32(const std::vector<uint32_t>& input, const std::vector<uint8_t>& bit_counts, std::vector<uint32_t>& output, size_t original_size) {
    size_t full_blocks = (original_size + 31) / 32;  // Round up
    output.resize(full_blocks * 32, 0);  // Initialize with zeros

    int input_offset = 0;
    for (size_t i = 0; i < full_blocks; ++i) {
        const int bits = bit_counts[i];
        if (bits == 0) {
            memset(output.data() + i * 32, 0, 32 * sizeof(uint32_t));
            continue;
        }
        cpu_deinterleave_bits(input.data() + input_offset, output.data() + i * 32, 1, bits);
        input_offset += bits;
    }

    // Truncate to original size
    output.resize(original_size);
}


//------------------------------------------------------------------------------
// Grouped Interleave Fixed-Size Functions

void interleave_to_32(const std::vector<uint32_t>& input, std::vector<uint32_t>& output) {
    size_t input_size = input.size();
    size_t full_blocks = input_size / 32;
    size_t remaining = input_size % 32;
    
    size_t output_size = (full_blocks + (remaining > 0 ? 1 : 0)) * 32;
    output.resize(output_size, 0);  // Initialize with zeros

    cpu_interleave_bits(input.data(), output.data(), full_blocks, 32);

    // Process remaining elements
    if (remaining > 0) {
        std::vector<uint32_t> padded_input(32, 0);
        std::memcpy(padded_input.data(), input.data() + full_blocks * 32, remaining * sizeof(uint32_t));
        cpu_interleave_bits(padded_input.data(), output.data() + full_blocks * 32, 1, 32);
    }
}

void deinterleave_from_32(const std::vector<uint32_t>& input, std::vector<uint32_t>& output, size_t original_size) {
    size_t full_blocks = (original_size + 31) / 32;  // Round up
    output.resize(full_blocks * 32, 0);  // Initialize with zeros
    
    cpu_deinterleave_bits(input.data(), output.data(), full_blocks, 32);

    // Truncate to original size
    output.resize(original_size);
}

void interleave_groups(const std::vector<uint32_t>& input, std::vector<uint32_t>& output) {
    size_t group_size = 32;
    size_t input_size = input.size();
    size_t group_count = (input_size + group_size - 1) / group_size;

    output.resize(input_size);

#ifdef INTERLEAVE_ONLY_LOCAL
        for (size_t g = 0; g < group_count; g += INTERLEAVE_ONLY_LOCAL) {
            size_t local_group_count = group_count - g;
            if (local_group_count > INTERLEAVE_ONLY_LOCAL) {
                local_group_count = INTERLEAVE_ONLY_LOCAL;
            }
            for (size_t i = 0; i < group_size; ++i) {
                for (size_t lg = 0; lg < local_group_count; ++lg) {
                    size_t input_idx = (g + lg) * group_size + i;
                    size_t output_idx = i * local_group_count + lg + g * group_size;
                    if (input_idx < input_size) {
                        output[output_idx] = input[input_idx];
                    } else {
                        output[output_idx] = 0;  // Pad with zeros if needed
                    }
                }
            }
        }
#else
        for (size_t i = 0; i < group_size; ++i) {
            for (size_t g = 0; g < group_count; ++g) {
                size_t input_idx = g * group_size + i;
                size_t output_idx = i * group_count + g;
                if (input_idx < input_size) {
                    output[output_idx] = input[input_idx];
                } else {
                    output[output_idx] = 0;  // Pad with zeros if needed
                }
            }
        }
#endif
}

void deinterleave_groups(const std::vector<uint32_t>& input, std::vector<uint32_t>& output, size_t original_size) {
    size_t group_size = 32;
    size_t group_count = (original_size + group_size - 1) / group_size;

    output.resize(original_size);

#ifdef INTERLEAVE_ONLY_LOCAL
        for (size_t g = 0; g < group_count; g += INTERLEAVE_ONLY_LOCAL) {
            size_t local_group_count = group_count - g;
            if (local_group_count > INTERLEAVE_ONLY_LOCAL) {
                local_group_count = INTERLEAVE_ONLY_LOCAL;
            }
            for (size_t i = 0; i < group_size; ++i) {
                for (size_t lg = 0; lg < local_group_count; ++lg) {
                    size_t input_idx = i * local_group_count + lg + g * group_size;
                    size_t output_idx = (g + lg) * group_size + i;
                    if (output_idx < original_size) {
                        output[output_idx] = input[input_idx];
                    }
                }
            }
        }
#else
        for (size_t i = 0; i < group_size; ++i) {
            for (size_t g = 0; g < group_count; ++g) {
                size_t input_idx = i * group_count + g;
                size_t output_idx = g * group_size + i;
                if (output_idx < original_size) {
                    output[output_idx] = input[input_idx];
                }
            }
        }
#endif
}

// ZSTD compression function
std::vector<uint8_t> zstd_compress(const std::vector<uint8_t>& data, int compression_level) {
    size_t const max_dst_size = ZSTD_compressBound(data.size());
    std::vector<uint8_t> compressed_data(max_dst_size);

    size_t const compressed_size = ZSTD_compress(
        compressed_data.data(), max_dst_size,
        data.data(), data.size(),
        compression_level);

    if (ZSTD_isError(compressed_size)) {
        throw std::runtime_error("ZSTD compression failed");
    }

    compressed_data.resize(compressed_size);
    return compressed_data;
}

// ZSTD decompression function
std::vector<uint8_t> zstd_decompress(const std::vector<uint8_t>& compressed_data, size_t original_size) {
    std::vector<uint8_t> decompressed_data(original_size);

    size_t const decompressed_size = ZSTD_decompress(
        decompressed_data.data(), original_size,
        compressed_data.data(), compressed_data.size());

    if (ZSTD_isError(decompressed_size) || decompressed_size != original_size) {
        throw std::runtime_error("ZSTD decompression failed");
    }

    return decompressed_data;
}

// New function to compress all components
std::vector<uint8_t> compress_all_components(
    const std::vector<uint8_t>& bit_counts,
    const std::vector<uint8_t>& high_index,
    const std::vector<uint32_t>& interleaved,
    const std::vector<uint32_t>& interleaved_high_bits)
{
    // Concatenate all components
    std::vector<uint8_t> all_data;
    
    // Add bit_counts
    all_data.insert(all_data.end(), bit_counts.begin(), bit_counts.end());
    
    // Add high_index
    all_data.insert(all_data.end(), high_index.begin(), high_index.end());
    
    // Add interleaved
    const uint8_t* interleaved_bytes = reinterpret_cast<const uint8_t*>(interleaved.data());
    all_data.insert(all_data.end(), interleaved_bytes, interleaved_bytes + interleaved.size() * sizeof(uint32_t));
    
    // Add interleaved_high_bits
    const uint8_t* interleaved_high_bits_bytes = reinterpret_cast<const uint8_t*>(interleaved_high_bits.data());
    all_data.insert(all_data.end(), interleaved_high_bits_bytes, interleaved_high_bits_bytes + interleaved_high_bits.size() * sizeof(uint32_t));
    
    // Compress all data
    return zstd_compress(all_data, ZSTD_COMPRESSION_LEVEL);
}

// New function to decompress all components
void decompress_all_components(
    const std::vector<uint8_t>& compressed_data,
    std::vector<uint8_t>& bit_counts,
    std::vector<uint8_t>& high_index,
    std::vector<uint32_t>& interleaved,
    std::vector<uint32_t>& interleaved_high_bits,
    size_t bit_counts_size,
    size_t high_index_size,
    size_t interleaved_size,
    size_t interleaved_high_bits_size)
{
    // Decompress all data
    std::vector<uint8_t> decompressed_data = zstd_decompress(compressed_data, 
        bit_counts_size + high_index_size + 
        interleaved_size * sizeof(uint32_t) + 
        interleaved_high_bits_size * sizeof(uint32_t));
    
    // Extract components
    size_t offset = 0;
    
    // Extract bit_counts
    bit_counts.assign(decompressed_data.begin(), decompressed_data.begin() + bit_counts_size);
    offset += bit_counts_size;
    
    // Extract high_index
    high_index.assign(decompressed_data.begin() + offset, decompressed_data.begin() + offset + high_index_size);
    offset += high_index_size;
    
    // Extract interleaved
    interleaved.resize(interleaved_size);
    std::memcpy(interleaved.data(), decompressed_data.data() + offset, interleaved_size * sizeof(uint32_t));
    offset += interleaved_size * sizeof(uint32_t);
    
    // Extract interleaved_high_bits
    interleaved_high_bits.resize(interleaved_high_bits_size);
    std::memcpy(interleaved_high_bits.data(), decompressed_data.data() + offset, interleaved_high_bits_size * sizeof(uint32_t));
}

void print_vector(const std::vector<uint32_t>& vec, const std::string& name, int max_size = 100) {
    std::cout << name << " (size " << vec.size() << "): ";
    for (size_t i = 0; i < vec.size() && i < max_size; ++i) {
        std::cout << std::setw(10) << vec[i] << " ";
    }
    if (vec.size() > max_size) std::cout << "...";
    std::cout << std::endl;
}

void print_vector(const std::vector<int32_t>& vec, const std::string& name, int max_size = 100) {
    std::cout << name << " (size " << vec.size() << "): ";
    for (size_t i = 0; i < vec.size() && i < max_size; ++i) {
        std::cout << std::setw(10) << vec[i] << " ";
    }
    if (vec.size() > max_size) std::cout << "...";
    std::cout << std::endl;
}

void print_vector(const std::vector<uint8_t>& vec, const std::string& name, int max_size = 100) {
    std::cout << name << " (size " << vec.size() << "): ";
    for (size_t i = 0; i < vec.size() && i < max_size; ++i) {
        std::cout << std::setw(3) << static_cast<int>(vec[i]) << " ";
    }
    if (vec.size() > max_size) std::cout << "...";
    std::cout << std::endl;
}

std::vector<int32_t> generate_gaussian_data(size_t n_samples, double mean = 78000.0, double std_dev = 317.0) {
    std::mt19937 gen{1337};
    std::normal_distribution<> dist{mean, std_dev};

    std::vector<int32_t> result;
    result.reserve(n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        double value = dist(gen);
        result.push_back(static_cast<int32_t>(std::round(value)));
    }

    return result;
}

// Modified test function
void test_compression() {
    std::vector<int32_t> original = generate_gaussian_data(10 * 1024 * 1024);
    const int num_blocks = (original.size() + QUANT_GROUP_SIZE - 1) / QUANT_GROUP_SIZE;

    std::vector<uint8_t> bit_counts;
    std::vector<uint8_t> high_index;
    std::vector<uint32_t> high_bits;
    std::vector<uint32_t> encoded_data;
    std::vector<uint32_t> interleaved, interleaved_high_bits;
    std::vector<uint32_t> group_interleaved, group_interleaved_high_bits;
    std::vector<uint8_t> compressed_all;

    compress_int32(original, bit_counts, high_index, high_bits, encoded_data);
#ifdef ENABLE_VARLEN_BITS
    interleave_to_32(encoded_data, bit_counts, interleaved);
#else
    interleave_to_32(encoded_data, interleaved);
#endif
#ifdef ENABLE_INTERLEAVE_HIGH_BITS
    interleave_to_32(high_bits, interleaved_high_bits);
#else
    interleaved_high_bits = high_bits;
#endif
#ifdef ENABLE_SECOND_PASS
    interleave_groups(interleaved, group_interleaved);
    interleave_groups(interleaved_high_bits, group_interleaved_high_bits);
#else
    group_interleaved = interleaved;
    group_interleaved_high_bits = interleaved_high_bits;
#endif

    encode_bit_counts(bit_counts);

    // Compress all components
    compressed_all = compress_all_components(bit_counts, high_index, group_interleaved, group_interleaved_high_bits);

    // --- This is where the data would be sent over the wire. ---

    print_vector(original, "original");
    print_vector(bit_counts, "bit_counts");
    print_vector(high_bits, "high_bits");
    print_vector(interleaved_high_bits, "interleaved_high_bits");
    print_vector(encoded_data, "encoded_data");
    print_vector(interleaved, "interleaved");
    print_vector(compressed_all, "compressed_all");

    // Decompress all components
    std::vector<uint8_t> decompressed_bit_counts;
    std::vector<uint8_t> decompressed_high_index;
    std::vector<uint32_t> decompressed_group_interleaved;
    std::vector<uint32_t> decompressed_group_interleaved_high_bits;
    std::vector<uint32_t> deinterleaved, deinterleaved_high_bits;
    std::vector<int32_t> decompressed;
    std::vector<uint32_t> decompressed_interleaved;
    std::vector<uint32_t> decompressed_interleaved_high_bits;

    decompress_all_components(
        compressed_all,
        decompressed_bit_counts,
        decompressed_high_index,
        decompressed_group_interleaved,
        decompressed_group_interleaved_high_bits,
        bit_counts.size(),
        high_index.size(),
        interleaved.size(),
        interleaved_high_bits.size()
    );

    decode_bit_counts(decompressed_bit_counts);

#ifdef ENABLE_SECOND_PASS
    deinterleave_groups(decompressed_group_interleaved, decompressed_interleaved, encoded_data.size());
    deinterleave_groups(decompressed_group_interleaved_high_bits, decompressed_interleaved_high_bits, high_bits.size());
#else
    decompressed_interleaved = decompressed_group_interleaved;
    decompressed_interleaved_high_bits = decompressed_group_interleaved_high_bits;
#endif
#ifdef ENABLE_INTERLEAVE_HIGH_BITS
    deinterleave_from_32(decompressed_interleaved_high_bits, deinterleaved_high_bits, high_bits.size());
#else
    deinterleaved_high_bits = decompressed_interleaved_high_bits;
#endif
#ifdef ENABLE_VARLEN_BITS
    deinterleave_from_32(decompressed_interleaved, decompressed_bit_counts, deinterleaved, encoded_data.size());
#else
    deinterleave_from_32(decompressed_interleaved, deinterleaved, encoded_data.size());
#endif
    decompress_int32(num_blocks, decompressed_bit_counts, decompressed_high_index, deinterleaved_high_bits, deinterleaved, decompressed);

    if (original == decompressed) {
        std::cout << "Compression test passed!" << std::endl;
    } else {
        std::cout << "Compression test failed!" << std::endl;
        return;
    }

    // Print compression ratio
    double compression_ratio = static_cast<double>(original.size() * sizeof(int32_t)) / compressed_all.size();
    std::cout << "Compression ratio: " << compression_ratio << std::endl;

    std::cout << "Number of blocks: " << num_blocks << std::endl;
}

int main() {
    test_compression();
    return 0;
}

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CUBPrefixSumTest CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architectures (adjust as needed for your GPU)
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)

# Add CUB include directory
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Add executable
add_executable(cub_prefix_sum_test test.cu)

# Set CUDA specific flags
set_target_properties(cub_prefix_sum_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optionally, you can add compiler flags
if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(cub_prefix_sum_test PRIVATE -Wall -Wextra -pedantic)
endif()

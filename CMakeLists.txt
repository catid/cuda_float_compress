cmake_minimum_required(VERSION 3.21)
project(cuda_float_compress LANGUAGES CXX CUDA)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "" FORCE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /Zi /O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -flto=auto")
    # -fsanitize=address
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -march=native")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -g -O2 -march=native")
endif()

# For pybind11 to work on Linux.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(Python3 3.7 COMPONENTS Interpreter Development NumPy)

# Link to Torch
if(NOT TARGET torch_library)
    execute_process(
    COMMAND
        ${Python3_EXECUTABLE} -c
        "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TORCH_PATH)
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})
 
    # Find PyTorch installation
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(f'{torch.utils.cmake_prefix_path}/Torch')"
        OUTPUT_VARIABLE Torch_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NOT Torch_DIR)
        message(FATAL_ERROR "PyTorch installation not found. Make sure it's installed in the current Python environment.")
    endif()
    message(STATUS "Found PyTorch installation: ${Torch_DIR}")

    # Note there is a noisy warning for missing kineto here
    # https://github.com/pytorch/pytorch/issues/62588
    find_package(Torch REQUIRED)

    find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
endif()

add_subdirectory(pybind11)

set(ZSTD_BUILD_PROGRAMS OFF)
set(ZSTD_BUILD_TESTS OFF)
set(ZSTD_BUILD_STATIC ON)
set(ZSTD_BUILD_SHARED OFF)
add_subdirectory(zstd/build/cmake EXCLUDE_FROM_ALL)

# Configure CUDA after finding Torch to avoid conflicts
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD "17")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CMAKE_CUDA_FLAGS_INIT "-std=c++17 -allow-unsupported-compiler")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -g -O2")
set(CMAKE_CUDA_ARCHITECTURES all-major)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -Werror -Xptxas --warn-on-spills")

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0")
    message(STATUS "Setting CUDA debug flags")
endif()

pybind11_add_module(cuda_float_compress
    src/cuda_float_compress.cpp
    src/cuszplus_f32.cu
)

target_include_directories(cuda_float_compress PUBLIC
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)
target_link_libraries(cuda_float_compress PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    Python3::NumPy
    Threads::Threads
    pybind11::module
    CUDA::cudart
    libzstd_static
)
set_target_properties(cuda_float_compress PROPERTIES PREFIX "") # remove lib prefix

set_target_properties(cuda_float_compress PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
install(TARGETS cuda_float_compress
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .)

message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

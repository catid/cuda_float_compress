cmake_minimum_required(VERSION 3.21)
project(cuda_float_compress LANGUAGES CXX CUDA)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    # -fsanitize=address
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -march=native")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
endif()

# For pybind11 to work on Linux.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Dependencies
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(Python3 3.7 COMPONENTS Interpreter Development NumPy)

# Make sure we are compatible with torch ABI. Must go before any
# add_subdirectory.
execute_process(
    COMMAND python -c "import torch; import os; print(int(torch._C._GLIBCXX_USE_CXX11_ABI), end='')"
    OUTPUT_VARIABLE TorchAbi
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${TorchAbi})

add_subdirectory(pybind11)

# Link to Torch
if(NOT TARGET torch_library)
    execute_process(
    COMMAND
        ${Python3_EXECUTABLE} -c
        "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TORCH_PATH)
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86)
    endif()
 
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
endif()

find_package(CUDAToolkit REQUIRED)

set(CUSZP_BUILD_EXAMPLES OFF)
add_subdirectory(cuSZp)

pybind11_add_module(cuda_float_compress cuda_float_compress.cpp)

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

target_include_directories(cuda_float_compress PUBLIC
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)
target_link_libraries(cuda_float_compress PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    cuSZp
    Python3::NumPy
    CUDA::cudart_static
    Threads::Threads
    pybind11::module
)
set_target_properties(cuda_float_compress PROPERTIES PREFIX "") # remove lib prefix

install(TARGETS cuda_float_compress DESTINATION .)

cmake_minimum_required(VERSION 3.21)
project(cuda_float_compress LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For pybind11 to work on Linux.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Dependencies
find_package(Threads REQUIRED)
find_package(Python3 3.7 COMPONENTS Interpreter Development NumPy)

# Make sure we are compatible with torch ABI. Must go before any
# add_subdirectory.
execute_process(
    COMMAND
        python -c
        "import torch; import os; print(int(torch._C._GLIBCXX_USE_CXX11_ABI), end='')"
        OUTPUT_VARIABLE TorchAbi)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${TorchAbi})

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# TORCH LIB  #####
if(NOT TARGET torch_library)
    execute_process(
    COMMAND
        python -c
        "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TORCH_PATH)
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})

    # If you want to target specific CUDA architectures, uncomment the line below
    #set(TORCH_CUDA_ARCH_LIST "6.0;7.0;7.5;8.0;8.6")

    # Note there is a noisy warning for missing kineto here
    # https://github.com/pytorch/pytorch/issues/62588
    find_package(Torch REQUIRED)
endif()

add_subdirectory(cuSZp)

pybind11_add_module(cuda_float_compress cuda_float_compress.cpp)

target_include_directories(cuda_float_compress PUBLIC
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)
target_link_libraries(cuda_float_compress PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARIES}
    cuSZp
    Python3::NumPy
)

install(TARGETS cuda_float_compress DESTINATION .)

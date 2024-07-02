cmake_minimum_required(VERSION 3.15)
project(cuda_float_compress LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)

add_subdirectory(cuSZp)

pybind11_add_module(cuda_float_compress cuda_float_compress.cpp)

target_link_libraries(cuda_float_compress PRIVATE
    ${TORCH_LIBRARIES}
    cuSZp
)

install(TARGET cuda_float_compress DESTINATION .)

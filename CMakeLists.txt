cmake_minimum_required(VERSION 3.18)
project(Zero2_Prototype LANGUAGES CXX CUDA)


function(print_compiler_info)
    message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Version: ${CUDA_VERSION_STRING}")
    message(STATUS "MPI CXX Compiler: ${MPI_CXX_COMPILER}")
endfunction()

set(CMAKE_CUDA_ARCHITECTURES 80)
message(STATUS "Targeting CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.4)
message(STATUS "Hinting CUDA Toolkit Root: ${CUDA_TOOLKIT_ROOT_DIR}")

find_package(CUDA REQUIRED)

find_package(MPI REQUIRED)


find_path(NCCL_INCLUDE_DIR nccl.h
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/include /usr/include /usr/local/include)

find_library(NCCL_LIBRARY NAMES nccl
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/lib)

if (NOT NCCL_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find nccl.h. Please ensure NCCL development package is installed.")
endif()
if (NOT NCCL_LIBRARY)
    message(FATAL_ERROR "Could not find libnccl.so. Please ensure NCCL package is installed.")
else()
    set(NCCL_FOUND TRUE)
    message(STATUS "Found NCCL Include Dir: ${NCCL_INCLUDE_DIR}")
    message(STATUS "Found NCCL Library: ${NCCL_LIBRARY}")
endif()


add_executable(zero2_demo
    main.cu
    utils.cu
    model.cu
    zero_optimizer.cu
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


target_include_directories(zero2_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${MPI_INCLUDE_PATH}
    ${CUDA_INCLUDE_DIRS}
    ${NCCL_INCLUDE_DIR} #
)

target_link_libraries(zero2_demo PRIVATE
    ${CUDA_LIBRARIES}
    MPI::MPI_CXX
    ${NCCL_LIBRARY} 
)


set(CMAKE_VERBOSE_MAKEFILE ON)


print_compiler_info()


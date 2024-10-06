cmake_minimum_required(VERSION 3.0.0)
project(torch_lab VERSION 0.1.0 LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -Wno-format-extra-args)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
include(utils)

set(CUDA_ARCH "80" CACHE STRING "CUDA archs")

# cutlass args
set(CUTLASS_ROOT ${CUTLASS_ROOT})
if(NOT CUTLASS_ROOT)
    message(FATAL_ERROR "CUTLASS_ROOT is required")
endif()

# /torch_install_dir/share/cmake/Torch
set(Torch_DIR ${Torch_DIR})
if(NOT Torch_DIR)
    message(FATAL_ERROR "Torch_DIR is required")
endif()

find_package(Torch REQUIRED)

# cutlass settings
set(CUTLASS_INCLUDE_DIRS 
    ${CUTLASS_ROOT}/include
    ${CUTLASS_ROOT}/tools/util/include
)

file(GLOB_RECURSE srcs 
    ${PROJECT_SOURCE_DIR}/*.cc
    ${PROJECT_SOURCE_DIR}/*.cu
)

add_library(torch_lab SHARED ${srcs})

target_include_directories(torch_lab PRIVATE 
    ${TORCH_INCLUDE_DIRS}
    ${CUTLASS_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(torch_lab 
    cublas_static 
    cublas 
    cublasLt
    ${TORCH_LIBRARIES}
)


# cuda arch check
set(ENABLE_90a FALSE) 
list_contains(CUDA_ARCH "90a" ENABLE_90a)
if(ENABLE_90a)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_90a")
endif()


target_link_libraries(torch_lab 
    ${TORCH_LIBRARIES}
)

set_target_properties(torch_lab PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})


target_compile_options(torch_lab PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>
    -Wno-format-extra-args
    -Wno-format-overflow
)
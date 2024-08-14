
#pragma once
#include <cuda.h>
#include <iostream>
#include <stdio.h>

#pragma once

#define RUNTIME_ASSERT(expr,msg)                \
  do {                                          \
    if (!(expr)) throw std::runtime_error(msg); \
  } while (0)                                   

#define UP_DIV(x, y) (((x) + (y)-1) / (y))
#define UP_ROUND(x, y) ((((x) + (y)-1) / (y)) * (y))
#define DOWN_ROUND(x, y) (((x) / (y)) * (y))

#define FULL_MASK 0xFFFFFFFF
#define WARP_SIZE 32

#define CUDA_ERROR_CHECK(call)                                                 \
  do {                                                                         \
    const cudaError_t error_code = call;                                       \
    if (error_code != cudaSuccess) {                                           \
      printf("CUDA Error:\n");                                                 \
      printf("    File:   %s\n", __FILE__);                                    \
      printf("    Line:   %d\n", __LINE__);                                    \
      printf("    Error:  %s\n", cudaGetErrorString(error_code));              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)


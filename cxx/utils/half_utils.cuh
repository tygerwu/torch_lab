#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace LAB {

namespace CUDA {


// Half Utils
template<typename T>
struct H2Traits{};

template<>
struct H2Traits<half>{
    using Type = __half2;
};

template<>
struct H2Traits<__nv_bfloat16>{
    using Type = __nv_bfloat162;
};


template<typename H2>
__device__ float2 H2ToF2(const H2& h2){
    if constexpr(std::is_same_v<H2,__half2>){
        return __float22half2_rn(f2);
    }
    if constexpr(std::is_same_v<H2,__nv_bfloat162>){
        return __float22bfloat162_rn(f2);
    }
}

template<typename H2>
__device__ H2 F2ToH2(const float2& f2){
    if constexpr(std::is_same_v<H2,__half2>){
        return __half22half2(f2);
    }
    if constexpr(std::is_same_v<H2,__nv_bfloat162>){
        return __bfloat162bfloat162(f2);
    }
}



}
}
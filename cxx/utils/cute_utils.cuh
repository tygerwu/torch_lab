#pragma once
#include "cute/tensor.hpp" 
#include <cuda.h>


namespace LAB{

namespace CUDA{

template<typename T>
CUTE_HOST_DEVICE 
void Print(const char* msg,const T& obj){
    cute::print(msg);
    cute::print(obj);
    cute::print("\n");
}


template<typename T>
CUTE_HOST_DEVICE
void PrintIden(const char* msg,const T& obj){
    cute::print(msg);
    cute::print(obj);
    cute::print("\n");

    for(int i=0; i<size(obj); i++){
        cute::print(obj(i));
        cute::print(" ");
    }
    cute::print("\n");
}


template<typename T>
CUTE_HOST_DEVICE
void PrintValue(const char* msg,const T& obj){
    cute::print(msg);
    cute::print(obj);
    cute::print("\n");

    for(int i=0; i<size(obj); i++){
       printf("%6.1f",static_cast<float>(obj(i)));
    }
    cute::print("\n");
}

// Arrange [B,R-E) into one group
template<int B,int E,class SrcEngine,class SrcLayout>
CUTE_HOST_DEVICE static constexpr auto 
group_diff(const cute::Tensor<SrcEngine,SrcLayout>& tensor) {
    constexpr int R = SrcLayout::rank;
    return cute::group_modes<B,R-E>(tensor);
}


}
}


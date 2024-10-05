#pragma once
#include "params.cuh"

namespace LAB{
namespace CUDA{

namespace SM8X{

template<typename T>
void AttentionInferV1(const AttnInferParams& params,cudaStream_t stream);

template<typename T>
void AttentionInferV2(const AttnInferParams& params,cudaStream_t strea);

}
}
}
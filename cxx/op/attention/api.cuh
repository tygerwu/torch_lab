#pragma once
#include "params.cuh"

namespace LAB{
namespace CUDA{

namespace SM80{

template<typename T>
void AttentionInferV1(const AttnInferParams& params,cudaStream_t stream);


}
}
}
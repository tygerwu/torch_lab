#pragma once
#include <math.h>

namespace LAB{
namespace CUDA{


struct AttnInferParams{
    // Pointers
    const void* __restrict__ q_ptr;
    const void* __restrict__ k_ptr;
    const void* __restrict__ v_ptr;
    void* o_ptr;

    // Stride
    int64_t q_batch_stride = 0;
    int64_t k_batch_stride = 0;
    int64_t v_batch_stride = 0;
    int64_t o_batch_stride = 0;

    int64_t q_seq_stride = 0;
    int64_t k_seq_stride = 0;
    int64_t v_seq_stride = 0;
    int64_t o_seq_stride = 0;

    int64_t q_head_stride = 0;
    int64_t k_head_stride = 0;
    int64_t v_head_stride = 0;
    int64_t o_head_stride = 0;

    // Shape 
    int batch,head_num,head_dim;
    int qo_seqlen,kv_seqlen;

    float softmax_scale;
    float log2_scale;
};


}
}
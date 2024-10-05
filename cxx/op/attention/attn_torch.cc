#include "utils/torch_utils.cuh"
#include "params.cuh"
#include "api.cuh"
#include "cute/tensor.hpp"

torch::Tensor mha_infer(const torch::Tensor& q,  // batch,qo_seqlen,head_num,head_size
                        const torch::Tensor& k,
                        const torch::Tensor& v,  // batch,kv_seqlen,head_num,head_size
                        double softmax_scale,
                        std::string backend) {
    // Check Dtype
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16, "q dtype must be float16 or bfloat16");
    TORCH_CHECK(k.dtype() == q_dtype && v.dtype() == q_dtype, "k,v dtype must be same as q");

    // Check Device
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm80 = (dprops->major == 8);
    TORCH_CHECK(is_sm80, "mha_infer only support sm80");
    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);

    // Check Layout
    CHECK_CONTIGUOUS(q)
    CHECK_CONTIGUOUS(k)
    CHECK_CONTIGUOUS(v)

    // Shapes
    const auto& q_shape = q.sizes();
    const auto& v_shape = v.sizes();

    int batch = q_shape[0];
    int qo_seqlen = q_shape[1];
    int head_num = q_shape[2];
    int head_size = q_shape[3];
    int kv_seqlen = v_shape[1];

    LAB::CUDA::AttnInferParams params;

    params.batch = batch;
    params.qo_seqlen = qo_seqlen;
    params.head_num = head_num;
    params.head_dim = head_size;
    params.kv_seqlen = kv_seqlen;

    params.softmax_scale = softmax_scale;
    params.log2_scale = softmax_scale * M_LOG2E;

    params.q_head_stride = head_size;
    params.k_head_stride = head_size;
    params.v_head_stride = head_size;
    params.o_head_stride = head_size;

    params.q_seq_stride = head_size * head_num;
    params.k_seq_stride = head_size * head_num;
    params.v_seq_stride = head_size * head_num;
    params.o_seq_stride = head_size * head_num;

    params.q_batch_stride = head_size * head_num * qo_seqlen;
    params.o_batch_stride = head_size * head_num * qo_seqlen;
    params.k_batch_stride = head_size * head_num * kv_seqlen;
    params.v_batch_stride = head_size * head_num * kv_seqlen;
    // output
    auto o = torch::empty_like(q);

    // Ptrs
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = o.data_ptr();

    at::cuda::CUDAGuard device_gard(q.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (backend == "attn_infer_v1") {
        LAB::CUDA::SM8X::AttentionInferV1<cutlass::half_t>(params, stream);
    }else if(backend == "attn_infer_v2"){
        LAB::CUDA::SM8X::AttentionInferV2<cutlass::half_t>(params, stream);
    }

    return o;
}

TORCH_LIBRARY_FRAGMENT(torch_lab, m) {
    m.def("mha_infer", &mha_infer);
}
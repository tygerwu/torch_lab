import torch
from triton import ops
from attn_infer import byte_gemm_func


def attn_tflops(batch, qo_seq_len, kv_seq_len, head_num, head_dim):
    # qxk
    qk_flops = 2 * batch * head_dim * head_num * qo_seq_len * kv_seq_len
    # scale
    scale_flops = batch * head_num * qo_seq_len * kv_seq_len
    # softmax
    softmax_flops = batch * head_num * qo_seq_len * kv_seq_len
    # pxv
    pv_flops = 2 * batch * head_dim * head_num * qo_seq_len * kv_seq_len
    return (qk_flops + scale_flops + softmax_flops + pv_flops) / 1e12


# a:mxk
# b:kxn
def bmm(a, b, use_triton=False):
    if a.element_size() == 1:
        if len(a.size()) == 3 and len(b.size()) == 3:
            batch = a.size()[0]
            outputs = []
            for i in range(batch):
                outputs.append(byte_gemm_func(a[i, :, :], b[i, :, :]))
            return torch.stack(outputs)
        else:
            return byte_gemm_func(a, b)

    if use_triton:
        shape = a.size()
        batch = shape[0]
        m = shape[1]
        n = b.size()[2]
        c = torch.empty((batch, m, n), dtype=a.dtype, device=a.device)
        for i in range(batch):
            c[i, :, :] = ops.matmul(a[i, :, :], b[i, :, :])
        return c
    else:
        return torch.matmul(a, b)


def simple_flash_attn(q, k, v, softmax_scale, q_scale=None, k_scale=None):
    # q: qo_seq_len,HN,HD
    # qscale: qo_seq_len

    q_dtype = q.dtype
    q = q.permute(1, 0, 2).contiguous()  # HN,qo_seq_len,HD
    k = k.permute(1, 0, 2).contiguous()
    v = v.permute(1, 0, 2).contiguous()

    x = bmm(q, k.transpose(-2, -1).contiguous())
    x = x.to(torch.float)  # HN,qo_seq_len,kv_seq_len
    if q_scale is not None and k_scale is not None:
        x *= q_scale.view(1, -1, 1)
        x *= k_scale.view(1, 1, -1)

    p = torch.softmax(x * softmax_scale, dim=-1)
    p = p.to(v.dtype)

    o = bmm(p, v).to(v.dtype)

    o = o.permute(1, 0, 2).contiguous()
    return o


def torch_flash_attn(
    query,  # batch,qo_seq_len,HN,HD
    key,  # batch,kv_seq_len,HN,HD
    value,  # batch,kv_seq_len,HN,HD
    softmax_scale,
    q_scales=None,  # batch,hn,qo_seq_len(/BM)
    k_scales=None,
    per_block=False,
    BM=128,
    BN=128,
):
    q_dtype = query.dtype
    v_dtype = value.dtype
    q_shape = query.shape
    v_shape = value.shape
    device = query.device

    key = key.permute(0, 2, 3, 1).contiguous()  # batch,HN,HD,kv_seq_len

    BATCH = q_shape[0]
    QO_SEQ_LEN = q_shape[1]
    HN = q_shape[2]
    HD = q_shape[3]

    KV_SEQ_LEN = v_shape[1]

    PAD_QO_SEQ_LEN = ((QO_SEQ_LEN - 1) // BM + 1) * BM
    PAD_KV_SEQ_LEN = ((KV_SEQ_LEN - 1) // BN + 1) * BN

    out = torch.empty(q_shape, dtype=v_dtype, device=device)  # batch,qo_seq_len,HN,HD

    lse = torch.empty((BATCH, HN, QO_SEQ_LEN), device=device)

    for b in range(BATCH):
        for h in range(HN):

            # QO_SEQ_LEN x HD
            q_head = query[b, :, h, :]
            o_head = out[b, :, h, :]
            v_head = value[b, :, h, :]
            # HD x KV_SEQ_LEN
            k_head = key[b, h, :, :]

            # O = O x e(prev_max - g_max) + e(x-cur_max) v
            bm_idx = 0
            for bm in range(0, PAD_QO_SEQ_LEN, BM):
                R_BM = BM
                if bm + BM >= QO_SEQ_LEN:
                    R_BM = QO_SEQ_LEN - bm

                prev_sum = torch.zeros(R_BM, device=device)
                prev_max = torch.fill(torch.empty(R_BM, device=device), -float("inf"))

                # BM x HD
                q_block = q_head[bm : bm + R_BM, :].contiguous()

                # BM x HD
                o_block = torch.zeros((R_BM, HD), device=device)
                bn_idx = 0
                for bn in range(0, PAD_KV_SEQ_LEN, BN):
                    R_BN = BN
                    if bn + BN >= KV_SEQ_LEN:
                        R_BN = KV_SEQ_LEN - bn

                    # HD x BN
                    k_block = k_head[:, bn : bn + R_BN].contiguous()

                    # BM x BN
                    x_block = bmm(q_block, k_block).to(torch.float32)
                    # print("x_block:", x_block.contiguous())

                    if q_scales is not None and k_scales is not None:
                        if per_block:
                            q_scale = q_scales[b, h, bm_idx]
                            k_scale = k_scales[b, h, bn_idx]
                            x_block *= q_scale * k_scale
                        else:
                            q_scale_block = q_scales[b, h, bm : bm + R_BM]
                            k_scale_block = k_scales[b, h, bn : bn + R_BN]
                            x_block *= q_scale_block.view(-1, 1)
                            x_block *= k_scale_block.view(1, -1)

                    # BN x HD
                    v_block = v_head[bn : bn + R_BN, :].contiguous()

                    # compute g_max
                    cur_max = torch.max(x_block, dim=-1)[0]
                    g_max = torch.maximum(cur_max, prev_max)

                    # compute g_sum
                    cur_sum = torch.sum(
                        torch.exp((x_block - cur_max.view(-1, 1)) * softmax_scale),
                        dim=-1,
                    )
                    g_sum = prev_sum * torch.exp(
                        (prev_max - g_max) * softmax_scale
                    ) + cur_sum * torch.exp((cur_max - g_max) * softmax_scale)

                    # update o
                    o_scale = torch.exp((prev_max - g_max) * softmax_scale)
                    o_block *= o_scale.view(-1, 1)

                    # update x
                    p_block = torch.exp(
                        softmax_scale * (x_block - g_max.view(-1, 1))
                    ).to(v_dtype)
                    o_block += bmm(p_block, v_block).to(torch.float32)

                    # update params
                    prev_sum = g_sum
                    prev_max = g_max

                    bn_idx += 1

                # rescale
                o_block /= prev_sum.view(-1, 1)
                o_head[bm : bm + R_BM, :] = o_block.to(v_dtype)
                lse[b, h, bm : bm + R_BM] = prev_max * softmax_scale + torch.log(
                    prev_sum
                )

                bm_idx += 1

    return out, lse
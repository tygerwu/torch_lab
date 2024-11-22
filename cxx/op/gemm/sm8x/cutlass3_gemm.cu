

#include "gemm.cuh"
#include "cutlass3_ada_fp8_gemm_config.cuh"
#include "utils/helpers.cuh"
#include "utils/macro.cuh"

// MIDThrottle: CVT functions

namespace LAB {

namespace CUDA {

namespace SM8X {

template <typename T, int CFG_ID>
__global__ void Cutlass3GemmKernel(GemmParams params) {
    using CFG      = typename Cutlass3GemmConfigTrait<T, CFG_ID>::CFG;
    using BM       = typename CFG::BM;
    using BN       = typename CFG::BN;
    using BK       = typename CFG::BK;
    using DTypeD   = float;
    using BKSlices = typename CFG::BKSlices;
    using BKStages = typename CFG::BKStages;

    extern __shared__ char smem[];

    // Positions
    int tid   = threadIdx.x;
    int bm_id = blockIdx.x;
    int bn_id = blockIdx.y;

    // Instances
    auto tiled_g2s_a  = typename CFG::G2SCopyA{};
    auto tiled_g2s_b  = typename CFG::G2SCopyB{};
    auto tiled_ab_mma = typename CFG::ABMMA{};

    auto tiled_s2r_a = typename CFG::S2RCopyA{};
    auto tiled_s2r_b = typename CFG::S2RCopyB{};

    auto tiled_r2s_d = typename CFG::R2SCopyD{};
    auto tiled_s2g_d = typename CFG::S2GCopyD{};

    auto g2s_a = tiled_g2s_a.get_slice(tid);
    auto g2s_b = tiled_g2s_b.get_slice(tid);

    auto s2r_a = tiled_s2r_a.get_slice(tid);
    auto s2r_b = tiled_s2r_b.get_slice(tid);

    auto r2s_d = tiled_r2s_d.get_slice(tid);
    auto s2g_d = tiled_s2g_d.get_slice(tid);

    int M = params.M * params.B;
    int N = params.N;
    int K = params.K;

    auto ga_layout = make_layout(make_shape(M, K), make_stride(K, _1{}));
    auto gb_layout = make_layout(make_shape(N, K), make_stride(K, _1{}));
    auto gd_layout = make_layout(make_shape(M, N), make_stride(N, _1{}));

    auto ga = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(params.a_ptr)), ga_layout);
    auto gb = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(params.b_ptr)), gb_layout);
    auto gd = make_tensor(make_gmem_ptr(reinterpret_cast<DTypeD*>(params.d_ptr)), gd_layout);

    auto ga_block = local_tile(ga, make_shape(BM{}, BK{}), make_coord(bm_id, _));      // BM,BK,BKNum
    auto gb_block = local_tile(gb, make_shape(BN{}, BK{}), make_coord(bn_id, _));      // BN,BK,BKNum
    auto gd_block = local_tile(gd, make_shape(BM{}, BN{}), make_coord(bm_id, bn_id));  // BM,BN,BNNum

    // SMem B,V,D
    auto sa =
        make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem + typename CFG::SDffA{})), typename CFG::SMemLayoutA{});
    auto sb =
        make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem + typename CFG::SDffB{})), typename CFG::SMemLayoutB{});

    // G2S A,B
    auto g2s_src_a = g2s_a.partition_S(ga_block);  // (16,1),G2S_ValTile_BM,G2S_ValTile_BK,BKNum
    auto g2s_dst_a = g2s_a.partition_D(sa);        // (16,1),G2S_ValTile_BN,G2S_ValTile_BK,BKStages

    auto g2s_src_b = g2s_b.partition_S(gb_block);  // (16,1),G2S_ValTile_BN,G2S_ValTile_BK,BKNum
    auto g2s_dst_b = g2s_b.partition_D(sb);        // (16,1),G2S_ValTile_BN,G2S_ValTile_BK,BKStages

    // Reg A,B
    auto ra  = make_tensor<T>(typename CFG::RShapeA{});      // (4,2,2),ABMMA_ValTile_BM,ABMMA_ValTile_BK
    auto rb  = make_tensor<T>(typename CFG::RShapeB{});      // (4,2),  ABMMA_ValTile_BN,ABMMA_ValTile_BK
    auto acc = make_tensor<float>(typename CFG::RShapeD{});  // (2,2),  ABMMA_ValTile_BM,ABMMA_ValTile_BN
    clear(acc);

    // S2R A,B
    auto s2r_src_a = s2r_a.partition_S(sa);  // (16,1),S2R_ValTile_BM,S2R_ValTile_BK,BKStages
    auto s2r_dst_a = s2r_a.retile_D(ra);     // (16,1),S2R_ValTile_BM,S2R_ValTile_BK

    auto s2r_src_b = s2r_b.partition_S(sb);  // (16,1),S2R_ValTile_BN,S2R_ValTile_BK,BKStages
    auto s2r_dst_b = s2r_b.retile_D(rb);     // (16,1),S2R_ValTile_BN,S2R_ValTile_BK

    int gbk_num = params.K / BK{};
    // G2S Prefetch
    int gbk = 0;
    // Issuse N-1 cp_async_fence
    for (int i = 0; i < BKStages{} - 1; i++) {
        if (gbk < gbk_num) {
            copy(tiled_g2s_a, g2s_src_a(_, _, _, gbk), g2s_dst_a(_, _, _, gbk));
            copy(tiled_g2s_b, g2s_src_b(_, _, _, gbk), g2s_dst_b(_, _, _, gbk));
            ++gbk;
        }
        // Always Issue
        cp_async_fence();
    }
    int smem_read  = 0;
    int smem_write = gbk;

    // S2R Prefetch
    if (gbk > 0) {
        // Wait for 1st smem block
        cp_async_wait<BKStages{} - 2>();
        __syncthreads();

        // Prefetch for first MicroKernel
        copy(tiled_s2r_a, s2r_src_a(_, _, _0{}, _0{}), s2r_dst_a(_, _, _0{}));
        copy(tiled_s2r_b, s2r_src_b(_, _, _0{}, _0{}), s2r_dst_b(_, _, _0{}));
    }

    auto s2r_src_a_view = s2r_src_a(_, _, _, smem_read);
    auto s2r_src_b_view = s2r_src_b(_, _, _, smem_read);

    for (int bk = 0; bk < gbk_num; bk++) {
        // A MMATile consists of multiple MMASlices
        for_each(make_int_sequence<BKSlices{}>{}, [&](auto bk_slice) {
            // Last slice in BK
            if (bk_slice == BKSlices{} - 1) {
                // Wait SMemBlock for next MMATile
                cp_async_wait<BKStages{} - 2>();
                __syncthreads();
                // Update view for next MMATile
                // smem_read has been updated when bk_slice==0
                s2r_src_a_view = s2r_src_a(_, _, _, smem_read);
                s2r_src_b_view = s2r_src_b(_, _, _, smem_read);
            }
            // Prefetch for next MicroKernel in current or next Tile
            int next_bk_slice = (bk_slice + Int<1>{}) % BKSlices{};
            copy(tiled_s2r_a, s2r_src_a_view(_, _, next_bk_slice), s2r_dst_a(_, _, next_bk_slice));
            copy(tiled_s2r_b, s2r_src_b_view(_, _, next_bk_slice), s2r_dst_b(_, _, next_bk_slice));

            if (bk_slice == 0) {
                if (gbk < gbk_num) {
                    // Issue one more G2S if possable
                    copy(tiled_g2s_a, g2s_src_a(_, _, _, gbk), g2s_dst_a(_, _, _, smem_write));
                    copy(tiled_g2s_b, g2s_src_b(_, _, _, gbk), g2s_dst_b(_, _, _, smem_write));
                    ++gbk;
                    // Recycle
                    ++smem_write;
                    smem_write = (smem_write == BKStages{}) ? 0 : smem_write;
                }
                cp_async_fence();  // Always Issue
                // Recycle
                ++smem_read;
                smem_read = (smem_read == BKStages{}) ? 0 : smem_read;
            }

            cute::gemm(tiled_ab_mma, acc, ra(_, _, bk_slice), rb(_, _, bk_slice), acc);
        });
    }

    // Do mask for K

    auto sd = make_tensor(make_smem_ptr(reinterpret_cast<DTypeD*>(smem + typename CFG::SDffD{})),
                          typename CFG::SMemLayoutD{}(_, _, _0{}));

    // Reg HD

    auto r2s_src_d = group_diff<1, 0>(flatten(r2s_d.retile_S(acc)));    // ((2),S2RAtom_ValTile_PVMMA_M,S2RAtom_ValTile_PVMMA_N,S2R_ValTile_BM,S2R_ValTile_BN2,BN2Num
    auto r2s_dst_d = group_diff<1, 0>(flatten(r2s_d.partition_D(sd)));  // ~

    auto s2g_src_d = s2g_d.partition_S(sd);        // (8,1),S2G_ValeTile_BM,S2G_ValeTile_BN2,BN2Num
    auto s2g_dst_d = s2g_d.partition_D(gd_block);  // ~

    // r2s
    copy(tiled_r2s_d, r2s_src_d, r2s_dst_d);
    __syncthreads();
    // s2g
    copy(tiled_s2g_d, s2g_src_d, s2g_dst_d);
}

template <typename T>
static void LaunchGemm(const GemmParams& params) {
    static constexpr int CFG_ID = 0;

    using CFG = typename Cutlass3GemmConfigTrait<T, CFG_ID>::CFG;
    using BM  = typename CFG::BM;
    using BN  = typename CFG::BN;
    using BK  = typename CFG::BK;

#ifndef __CUDA_ARCH__
    // CFG{}.print();
#endif

    int M = params.B * params.M;
    int N = params.N;
    int K = params.K;

    RUNTIME_ASSERT(M % BM{} == 0, "Unsupported M");
    RUNTIME_ASSERT(N % BN{} == 0, "Unsupported N");
    RUNTIME_ASSERT(K % BK{} == 0, "Unsupported K");

    auto func       = Cutlass2GemmKernel<T, CFG_ID>;
    int  threads    = typename CFG::Threads{};
    int  smem_bytes = CFG::SBytes;
    if (smem_bytes >= (48 << 10)) {
        CUDA_ERROR_CHECK(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }

    dim3 grid(UP_DIV(M, BM{}), UP_DIV(N, BN{}));
    dim3 block(threads);

    func<<<grid, block, smem_bytes, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

#ifdef LAB_ENABLE_FP8
template <>
void Cutlass3Gemm<cutlass::float_e4m3_t>(const GemmParams& params) {
    LaunchGemm<cutlass::float_e4m3_t>(params);
}
#endif

}  // namespace SM8X
}  // namespace CUDA
}  // namespace LAB
#pragma once
#include "cute/tensor.hpp"
#include <cuda.h>

namespace LAB {

namespace CUDA {

using namespace cute;

enum class Pred2DAxis { M, N, MN };

template <int M, int N, Pred2DAxis PredAxis>
struct Pred2DStrideTrait {};

template <int M, int N>
struct Pred2DStrideTrait<M, N, Pred2DAxis::M> {
    using Stride = Stride<_1, _0>;
};

template <int M, int N>
struct Pred2DStrideTrait<M, N, Pred2DAxis::N> {
    using Stride = Stride<_0, _1>;
};

template <int M, int N>
struct Pred2DStrideTrait<M, N, Pred2DAxis::MN> {
    using Stride = Stride<_1, Int<M>>;
};

// G2S or S2G with Mask
template <Pred2DAxis PredAxis, int M, int N,
          typename TiledCopy, typename SrcTensor, typename DstTensor>
CUTE_DEVICE 
void SMem2DCopyWithMasK(const TiledCopy& tiled_copy, const SrcTensor& src, DstTensor& dst, 
                        int slice_id,int m_beg, int m_end, int n_beg, int n_end) {
    using PredStride = typename Pred2DStrideTrait<M, N, PredAxis>::Stride;
    using MNShape = Shape<Int<M>, Int<N>>;
    auto pred = make_tensor<bool>(MNShape{}, PredStride{});

    // Identity Layout
    auto iden = make_identity_tensor(MNShape{});
    auto thr_iden = tiled_copy.get_slice(slice_id).partition_S(iden);   // Ex:(CopyAtomSize),G2S_ValTile_M,G2S_ValTile_N

    static_assert(rank(thr_iden) == 3, "Invalid thr_iden");

    // Boundary Check
    if constexpr (PredAxis == Pred2DAxis::M) {
        for (int i = 0; i < M; i++) {
            pred(i, _0{}) = (get<0>(thr_iden(_0{}, i, _0{})) + m_beg < m_end);
        }
    } else if constexpr (PredAxis == Pred2DAxis::N) {
        for (int i = 0; i < N; i++) {
            pred(_0{}, i) = (get<0>(thr_iden(_0{}, _0{}, i)) + n_beg < n_end);
        }
    } else if constexpr (PredAxis == Pred2DAxis::MN) {
        for (int i = 0; i < M; i++) {
            int m_off = get<0>(thr_iden(_0{}, i, _0{}));
            bool m_mask = m_beg + m_off < m_end;
            for (int j = 0; j < N; j++) {
                int n_off = get<0>(thr_iden(_0{}, _0{}, j));
                bool n_mask = n_beg + n_off < n_end;
                pred(i, j) = m_mask && n_mask;
            }
        }
    }
    copy_if(tiled_copy, pred, src, dst);
}

}  // namespace CUDA
}  // namespace LAB

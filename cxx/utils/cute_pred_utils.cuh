#pragma once
#include "cute/tensor.hpp"
#include <cuda.h>

namespace LAB {

namespace CUDA {

namespace SM8X{

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
template <Pred2DAxis PredAxis, int SM, int SN,int ThrPredM, int ThrPredN,
          typename TiledCopy, typename PredTensor>
CUTE_DEVICE 
void Fill2DPred(const TiledCopy& tiled_copy,PredTensor& thr_pred,
                int slice_id,int m_beg, int m_end, int n_beg, int n_end){
                        
    // Identity Layout
    auto iden = make_identity_tensor(Shape<Int<SM>, Int<SN>>{});
    auto thr_iden = tiled_copy.get_slice(slice_id).partition_S(iden);   // Ex:(CopyAtomSize),G2S_ValTile_M,G2S_ValTile_N
    
    static_assert(rank(thr_iden) == 3, "Invalid thr_iden");

    // Boundary Check
    if constexpr (PredAxis == Pred2DAxis::M) {
        for (int i = 0; i < ThrPredM; i++) {
            thr_pred(i, _0{}) = (get<0>(thr_iden(_0{}, i, _0{})) + m_beg < m_end);
        }
    } else if constexpr (PredAxis == Pred2DAxis::N) {
        for (int i = 0; i < ThrPredN; i++) {
            thr_pred(_0{}, i) = (get<1>(thr_iden(_0{}, _0{}, i)) + n_beg < n_end);
        }
    } else if constexpr (PredAxis == Pred2DAxis::MN) {
        for (int i = 0; i < ThrPredM; i++) {
            int m_off = get<0>(thr_iden(_0{}, i, _0{}));
            bool m_mask = m_beg + m_off < m_end;
            for (int j = 0; j < ThrPredN; j++) {
                int n_off = get<1>(thr_iden(_0{}, _0{}, j));
                bool n_mask = n_beg + n_off < n_end;
                thr_pred(i, j) = m_mask && n_mask;
            }
        }
    }
}

// UseCopyFence: Submit a cp_async_fence() after each 2dCopy.
// ThrSrcLayout: (CopyAtomSize),ValTile_M,VTile_N,...
template <bool UseCopyFence,
          typename TiledCopy,    typename PredTensor,
          typename ThrSrcEngine, typename ThrSrcLayout,
          typename ThrDstEngine, typename ThrDstLayout>
CUTE_HOST_DEVICE
void CopyIf2D(const TiledCopy& tiled_copy,
              const PredTensor& thr_pred,
              const Tensor<ThrSrcEngine,ThrSrcLayout>& thr_src, 
                    Tensor<ThrDstEngine,ThrDstLayout>& thr_dst){
    constexpr int SrcRank = rank(ThrSrcLayout{});      
    constexpr int DstRank = rank(ThrDstLayout{});      
    if constexpr (SrcRank > 3 && DstRank > 3){
        auto thr_src_view = group_modes<3,SrcRank>(thr_src);
        auto thr_dst_view = group_modes<3,DstRank>(thr_dst);
        constexpr int num = size<3>(group<3,SrcRank>(ThrSrcLayout{}));
        CUTE_UNROLL
        for(int i=0; i<num; i++){
            copy_if(tiled_copy, thr_pred, thr_src_view(_,_,_,i), thr_dst_view(_,_,_,i));
            if constexpr (UseCopyFence) {
                cp_async_fence();
            }
        }
    }else{ 
        copy_if(tiled_copy, thr_pred, thr_src, thr_dst);
        if constexpr (UseCopyFence) {
            cp_async_fence();
        }
    }
}



// G2S or S2G with Mask
template <Pred2DAxis PredAxis,int SM,int SN,bool UseCopyFence,
          typename TiledCopy, 
          typename ThrSrcEngine, typename ThrSrcLayout,
          typename ThrDstEngine, typename ThrDstLayout>
CUTE_DEVICE 
void SMem2DCopyWithMasK(const TiledCopy& tiled_copy, 
                        const Tensor<ThrSrcEngine,ThrSrcLayout>& thr_src, 
                              Tensor<ThrDstEngine,ThrDstLayout>& thr_dst, 
                        int slice_id,int m_beg, int m_end, int n_beg, int n_end) {

  

    // Src: (CopyAtomSize),G2S_ValTile_M,G2S_ValTile_N,.....
    constexpr int ThrPredM = size<1>(ThrSrcLayout{});
    constexpr int ThrPredN = size<2>(ThrDstLayout{});
    
    using PredStride = typename Pred2DStrideTrait<ThrPredM, ThrPredN, PredAxis>::Stride;
    
    auto thr_pred = make_tensor<bool>(Shape<Int<ThrPredM>,Int<ThrPredN>>{}, PredStride{});
    Fill2DPred<PredAxis,SM,SN,ThrPredM,ThrPredN>(tiled_copy,thr_pred,slice_id,m_beg,m_end,n_beg,n_end);
    CopyIf2D<UseCopyFence>(tiled_copy,thr_pred,thr_src,thr_dst);
   
}

template <Pred2DAxis PredAxis,bool UseCopyFence,typename TiledCopy, 
          typename SrcMemEngine,typename SrcMemLayout,
          typename DstMemEngine,typename DstMemLayout>
CUTE_DEVICE 
void SMem2DCopyWithMasK(const TiledCopy& tiled_copy, 
                        const Tensor<SrcMemEngine,SrcMemLayout>& src_block, 
                              Tensor<DstMemEngine,DstMemLayout>& dst_block, 
                        int slice_id,int m_beg, int m_end, int n_beg, int n_end) {
    
    static_assert(rank(SrcMemLayout{}) == rank(DstMemLayout{}));
     
    // SrcMemLayout:(BM,BN,~,~)
    constexpr int SM = size<0>(SrcMemLayout{});
    constexpr int SN = size<1>(SrcMemLayout{});
    
    auto thr_copy =  tiled_copy.get_slice(slice_id);
     
    auto s2g_src = thr_copy.partition_S(src_block);         // (CopyAtomSize),G2S_ValTile_M,G2S_ValTile_N
    auto s2g_dst = thr_copy.partition_D(dst_block);
    
    SMem2DCopyWithMasK<PredAxis,SM,SN,UseCopyFence>(tiled_copy, s2g_src, s2g_dst, slice_id, m_beg, m_end, n_beg, n_end);
}

}


}  // namespace CUDA
}  // namespace LAB


#pragma once
#include "cute/tensor.hpp"
#include "utils/macro.cuh"
#include "utils/cute_smem_utils.cuh"
#include "utils/cute_utils.cuh"


namespace LAB{
namespace CUDA{

namespace SM80{


using namespace cute;

/**
 *  Do pipelines along K 
 */

template<typename T,int HD,
         int TiledMMA_ThrTile_M,int TiledMMA_VaTile_M,int TiledMMA_VaTile_N,   
         int BKTileNum,int BK2TileNum,int BN2Size>
struct AttentionInferV1Config{

    using M_T = Int<int(4 * TiledMMA_ThrTile_M)>;
    using N_T = _1;
    using K_T = _1;
    
    // Mx16x16
    using QKMMA_M = Int<int(M_T{} * 1 * 16)>;
    using QKMMA_N = Int<int(N_T{} * 2 * 8)>;
    using QKMMA_K = Int<int(K_T{} * 1 * 16)>;
    
    using QKMMA = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                           Layout<Shape<M_T,N_T,K_T>>,
                           Tile<QKMMA_M,QKMMA_N,QKMMA_K>>;

    using Threads = Int<int(M_T{} * N_T{} * K_T{} * 32)>;

    // QK MMA
    using BM = Int<TiledMMA_VaTile_M * int(QKMMA_M{})>;
    using BN = Int<TiledMMA_VaTile_N * int(QKMMA_N{})>;
    using BK = Int<int(BKTileNum * QKMMA_K{})>;
    using BKNum    = Int<int(HD / BK{})>;             // G2S Pipeline Stages
    using BKTiles  = Int<BKTileNum>;                  // QKMMA Pipeline Stages
    using BK2Tiles = Int<BK2TileNum>;                 // PVMMA Pipeline Stages


    static_assert(HD % BK{} == 0, "Invalid BKTileNum");
    static_assert(HD % 16 == 0, "Invalid HD");

    // PV MMA
    using PVMMA = QKMMA;
    using BK2 = Int<int(BK2TileNum) * 16>;
    using BN2 = Int<int(BN2Size)>;
    using BK2Num = Int<int(BN{} / BK2{})>;
    using BN2Num = Int<int(HD   / BN2{})>;

    static_assert(BN{} % BK2{} == 0, "Invalid BN");
    static_assert(HD   % BN2{} == 0, "Invalid BN2");
    static_assert(BN2Num{} == 1, "Invalid BN2");

    // SMem Config
    using SMemConfigQ = HalfSMem::KMajorConfig<BK{},BM{},Threads{},T,BKNum>;                  // Pipeline along BK
    using SMemConfigK = HalfSMem::KMajorConfig<BK{},BN{},Threads{},T,BKNum>;
    using SMemConfigV = HalfSMem::MMajorConfig<BN2{},BK2{},Threads{},T,tuple<BN2Num,BK2Num>>; // Pipeline along BK2
    using SMemConfigO = HalfSMem::KMajorConfig<BN2{},BM{},Threads{},T,BN2Num>;

    // SMem Layout
    using SMemLayoutQ = typename SMemConfigQ::SMemLayout;       // (BM,BK,BKNum)
    using SMemLayoutK = typename SMemConfigK::SMemLayout;       // (BN,BK,BKNum)
    using SMemLayoutV = typename SMemConfigV::SMemLayout;       // (BN2,BK2,BN2Num,BK2Num)
    using SMemLayoutO = typename SMemConfigO::SMemLayout;       // (BM,BN2,BN2Num)

    // G2S
    using G2SCopyQ = typename SMemConfigQ::AsyncCopy;
    using G2SCopyK = typename SMemConfigK::AsyncCopy;
    using G2SCopyV = typename SMemConfigV::AsyncCopy;

    // S2R
    using S2RCopyAtom      = Copy_Atom<SM75_U32x4_LDSM_N,T>;
    using S2RTransCopyAtom = Copy_Atom<SM75_U16x8_LDSM_T,T>;

    using S2RCopyQ = decltype(make_tiled_copy_A(S2RCopyAtom{},QKMMA{}));
    using S2RCopyK = decltype(make_tiled_copy_B(S2RCopyAtom{},QKMMA{}));
    using S2RCopyV = decltype(make_tiled_copy_B(S2RTransCopyAtom{},PVMMA{}));

    // R2S
    using T2 = cutlass::AlignedArray<T,2>;
    using R2SCopyO = decltype(make_tiled_copy_C(Copy_Atom<UniversalCopy<T2>,T>{},PVMMA{}));

    // S2G
    using S2GCopyO = typename SMemConfigO::Copy;


    // SMem Size
    static constexpr int SBytes_Q = cosize(SMemLayoutQ{}) * sizeof(T);
    static constexpr int SBytes_K = cosize(SMemLayoutK{}) * sizeof(T);
    static constexpr int SBytes_V = cosize(SMemLayoutV{}) * sizeof(T);
    static constexpr int SBytes_O = cosize(SMemLayoutO{}) * sizeof(float);
    static constexpr int SBytes = cute::max(SBytes_Q+SBytes_K+SBytes_V,SBytes_O);

    // SMem Offsets
    //  <Q>,<K>,<V>SBytes
    //  <O>
    using SOffQ = _0; 
    using SOffK = Int<int(SOffQ{} + SBytes_Q)>;
    using SOffV = Int<int(SOffK{} + SBytes_K)>;
    using SOffO = SOffQ;

    // Register Shape
    using DoubleRegBufSize = Int<int(2*16)>;
    using RShapeQ = decltype(partition_shape_A(QKMMA{},Shape<BM,DoubleRegBufSize>{}));     // (2,2,2),QKMMA_ValTile_BM,2
    using RShapeK = decltype(partition_shape_B(QKMMA{},Shape<BN,DoubleRegBufSize>{}));     // (2,2),  QKMMA_ValTile_BN,2
    using RShapeX = decltype(partition_shape_C(QKMMA{},Shape<BM,BN>{}));                   // (2,2),  QKMMA_ValTile_BM,QKMMA_ValTile_BN

    using RShapeP = decltype(partition_shape_A(PVMMA{},Shape<BM,BN>{}));                   // (2,2,2),PVMMA_ValTile_BM,PVMMA_ValTile_BN
    using RShapeV = decltype(partition_shape_B(PVMMA{},Shape<BN2,DoubleRegBufSize>{}));
    using RShapeO = decltype(partition_shape_C(PVMMA{},Shape<BM,BN2,BN2Num>{}));           // Hold all data

    using RowsPT   = Int<int(TiledMMA_VaTile_M*2)>;
    using RShapeSM = Layout<Shape<RowsPT>>;


    using RSizeX = Int<size(RShapeX{})>;

#ifndef __CUDA_ARCH__
    void print(){
        printf("\n");
        Print("BM:",BM{});
        Print("BN:",BN{});
        Print("BK:",BK{});
        Print("BKNum:",BKNum{});
        Print("BKTiles:",BKTiles{});

        Print("BN2:",BN2{});
        Print("BK2:",BK2{});
        Print("BN2Num:",BN2Num{});
        Print("BK2Num:",BK2Num{});
        Print("BK2Tiles:",BK2Tiles{});

        Print("SMemBytes:",SBytes);


        Print("SMemLayoutQ:",SMemLayoutQ{});
        Print("SMemLayoutK:",SMemLayoutK{});
        Print("SMemLayoutV:",SMemLayoutV{});
        Print("SMemLayoutO:",SMemLayoutO{});

        Print("RShapeQ:",RShapeQ{});
        Print("RShapeK:",RShapeK{});
        Print("RShapeX:",RShapeX{});
        Print("RShapeP:",RShapeP{});
        Print("RShapeV:",RShapeV{});
        Print("RShapeO:",RShapeO{});


        Print("S2RCopyK:",S2RCopyK{});
        Print("S2RCopyV:",S2RCopyV{});
        Print("R2SCopyO:",R2SCopyO{});
    }
#endif


};



template<typename T,int HD>
struct AttentionInferV1ConfigTratis{
    static_assert("Invalid HD");
};


template<typename T>
struct AttentionInferV1ConfigTratis<T,64>{
    static constexpr int HD = 64;
    static constexpr int TiledMMA_ThrTile_M = 1;
    static constexpr int TiledMMA_VaTile_M  = 2;       
    static constexpr int TiledMMA_VaTile_N  = 8;        // BN = N x 16
    static constexpr int BKTileNum  = 4;                // BK = BKTileNum x 16
    static constexpr int BK2TileNum = 8;
    static constexpr int BN2Size = 64;


    using CFG = AttentionInferV1Config<T,HD,TiledMMA_ThrTile_M,TiledMMA_VaTile_M,TiledMMA_VaTile_N,BKTileNum,BK2TileNum,BN2Size>;
};


template<typename T>
struct AttentionInferV1ConfigTratis<T,128>{
    static constexpr int HD = 128;
    static constexpr int TiledMMA_ThrTile_M = 1;
    static constexpr int TiledMMA_VaTile_M  = 2;       
    static constexpr int TiledMMA_VaTile_N  = 4;        // BN = N x 16
    static constexpr int BKTileNum  = 2;                // BK = BKTileNum x 16
    static constexpr int BK2TileNum = 2;
    static constexpr int BN2Size = 128;


    using CFG = AttentionInferV1Config<T,HD,TiledMMA_ThrTile_M,TiledMMA_VaTile_M,TiledMMA_VaTile_N,BKTileNum,BK2TileNum,BN2Size>;
};



}
}
}
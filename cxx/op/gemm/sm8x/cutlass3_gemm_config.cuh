#pragma once
#pragma once
#include "cute/tensor.hpp"
#include "utils/cute_smem_utils.cuh"
#include "utils/ada_fp8_mma_trait.hpp"
#include "utils/print.cuh"
#include "cutlass3_gemm_utils.cuh"


namespace LAB {

namespace CUDA {

namespace SM8X {

/**
 * @brief Classical Gemm by Cute : MultiG2S stages and overlap between S2R and MMA
 */
using namespace cute;


template<typename T,
        int MTSize,int NTSize,int KTSize,
        int BMSize,int BNSize,int BKSize,
        int BKStageSize>
struct Cutlass3GemmConfig{
    using DTypeD = float;

    using M_T = Int<MTSize>;
    using N_T = Int<NTSize>;
    using K_T = Int<KTSize>;

    using BM = Int<BMSize>;
    using BN = Int<BNSize>;
    using BK = Int<BKSize>;
    using BKStages = Int<BKStageSize>;

    using MMA = typename MMATrait<T>::

    using MMA_M  = typename MMATrait<T>::MMA_M;
    using MMA_K  = typename MMATrait<T>::MMA_K;
    using MMA_N  = typename MMATrait<T>::MMA_N;
    using Atom   = typename MMATrait<T>::Atom;
    using DTypeD = typename MMATrait<T>::DTypeD;
    

    static_assert(BM{} % MMA_M{} == 0, "Invalid BM");
    static_assert(BN{} % MMA_N{} == 0, "Invalid BN");
    static_assert(BK{} % MMA_K{} == 0, "Invalid BK");


    using ABMMA = TiledMMA<MMA_Atom<Atom>,
                         Layout<Shape<M_T,N_T,K_T>>,
                                 Tile<MMA_M,MMA_N,MMA_K>>;

    using Threads = Int<int(M_T{} * N_T{} * K_T{} * 32)>;

    using BKSlices  = Int<int(BK{} / MMA_K{})>;            // Pipeline

    // SMem Config
    using SMemConfigA = SMem::KMajorConfig<BK{},BM{},Threads{},T,BKStages>;                  // Pipeline along BK
    using SMemConfigB = SMem::KMajorConfig<BK{},BN{},Threads{},T,BKStages>;
    using SMemConfigD = SMem::KMajorConfig<BN{},BM{},Threads{},DTypeD,Int<1>>;


    // SMem Layout
    using SMemLayoutA = typename SMemConfigA::SMemLayout;       // (BM,BK,BKStages)
    using SMemLayoutB = typename SMemConfigB::SMemLayout;       // (BN,BK,BKStages)
    using SMemLayoutD = typename SMemConfigD::SMemLayout;       // (BM,BN,1)

    // G2S
    using G2SCopyA = typename SMemConfigA::AsyncCopy;
    using G2SCopyB = typename SMemConfigB::AsyncCopy;

    // S2R
    using S2RCopyAtom      = Copy_Atom<SM75_U32x4_LDSM_N,T>;

    using S2RCopyA = decltype(make_tiled_copy_A(S2RCopyAtom{},ABMMA{}));
    using S2RCopyB = decltype(make_tiled_copy_B(S2RCopyAtom{},ABMMA{}));

    // R2S
    using D2 = cutlass::AlignedArray<DTypeD,2>;
    using R2SCopyD = decltype(make_tiled_copy_C(Copy_Atom<UniversalCopy<D2>,DTypeD>{},ABMMA{}));

    // S2G
    using S2GCopyD = typename SMemConfigD::Copy;


    // SMem Size
    static constexpr int SBytes_A = cosize(SMemLayoutA{}) * sizeof(T);
    static constexpr int SBytes_B = cosize(SMemLayoutB{}) * sizeof(T);
    static constexpr int SBytes_D = cosize(SMemLayoutD{}) * sizeof(DTypeD);
    static constexpr int SBytes = cute::max(SBytes_A+SBytes_B,SBytes_D);

    // SMem Dffsets
    //  <A>,<B>,<V>SBytes
    //  <D>
    using SDffA = _0; 
    using SDffB = Int<int(SDffA{} + SBytes_A)>;
    using SDffD = SDffA;

    // Register Shape
    using DoubleRegBufSize = Int<int(BKSlices{}*32)>;
    using RShapeA = decltype(partition_shape_A(ABMMA{},Shape<BM,BK>{}));     // (4,2,2),ABMMA_ValTile_BM,ABMMA_ValTile_BK i8
    using RShapeB = decltype(partition_shape_B(ABMMA{},Shape<BN,BK>{}));     // (4,2),  ABMMA_ValTile_BN,ABMMA_ValTile_BM i8
    using RShapeD = decltype(partition_shape_C(ABMMA{},Shape<BM,BN>{}));     // (2,2),  ABMMA_ValTile_BM,ABMMA_ValTile_BN i8


    // Dnly print on host
#ifndef __CUDA_ARCH__
    void print(){
        printf("\n");
        Print("BM:",BM{});
        Print("BN:",BN{});
        Print("BK:",BK{});
        Print("BKStages:",BKStages{});
        Print("BKSlices:",BKSlices{});


        Print("SMemBytes:",SBytes);

        Print("SMemLayoutA:",SMemLayoutA{});
        Print("SMemLayoutB:",SMemLayoutB{});
        Print("SMemLayoutD:",SMemLayoutD{});

        Print("RShapeA:",RShapeA{});
        Print("RShapeB:",RShapeB{});
        Print("RShapeD:",RShapeD{});

        Print("ABMMA:",ABMMA{});
        Print("G2SCopyA:",G2SCopyA{});
        Print("G2SCopyA Atom:",typename SMemConfigA::EPT{});
        Print("S2RCopyB:",S2RCopyB{});
        Print("R2SCopyD:",R2SCopyD{});
    }
#endif 
};


template<typename T,int CFG_ID>
struct Cutlass3GemmConfigTrait{
    static_assert("Invalid HD");
};


template<typename T>
struct Cutlass3GemmConfigTrait<T,0>{
    static constexpr int MTSize = 2;
    static constexpr int NTSize = 2;
    static constexpr int KTSize = 1;

    static constexpr int BMSize = 128;
    static constexpr int BNSize = 128;
    static constexpr int BKSize = 64;
    static constexpr int BKStageSize = 3;
    using CFG = Cutlass3GemmConfig<T,MTSize,NTSize,KTSize,BMSize,BNSize,BKSize,BKStageSize>;
};


}
}
}
#pragma once
#include "cute/tensor.hpp" 
#include <cuda.h>


namespace LAB{

namespace CUDA{

namespace HalfSMem{
    using namespace cute;

    // EPT : Elements per thread
    // Example:
    //   TotalElenum : 512
    //   Threads: 128
    //   Base = 8
    //   EPT = 4
    template<int Base,int TotalEleNum,int Threads>
    struct EPTTrait{
        // Check the value in descending order: 8,4,2,1
        using EPT = std::conditional_t<TotalEleNum % (Threads * Base) == 0,
                                        Int<Base>,
                                        typename EPTTrait<Base/2,TotalEleNum,Threads>::EPT  // Recursion
                                      >;
    };
    // Recursive border
    template<int TotalEleNum,int Threads>
    struct EPTTrait<0,TotalEleNum,Threads>{
        using EPT = _0;
    };

    
    template<int Cols,typename LogicalLayoutAtom>
    struct SMemLayoutAtomTrait{
        using SMemLayoutAtom = LogicalLayoutAtom;  // Data is naturally interleaved in smem. Ex:Cols=40
    };

    template<typename LogicalLayoutAtom>
    struct SMemLayoutAtomTrait<16,LogicalLayoutAtom>{
        using SMemLayoutAtom = decltype(composition(Swizzle<1,3,3>{},LogicalLayoutAtom{}));
    };

    template<typename LogicalLayoutAtom>
    struct SMemLayoutAtomTrait<32,LogicalLayoutAtom>{
        using SMemLayoutAtom = decltype(composition(Swizzle<2,3,3>{},LogicalLayoutAtom{}));
    };

    template<typename LogicalLayoutAtom>
    struct SMemLayoutAtomTrait<64,LogicalLayoutAtom>{
        using SMemLayoutAtom = decltype(composition(Swizzle<3,3,3>{},LogicalLayoutAtom{}));
    };

    template<typename LogicalLayoutAtom>
    struct SMemLayoutAtomTrait<128,LogicalLayoutAtom>{
        using SMemLayoutAtom = decltype(composition(Swizzle<3,3,4>{},LogicalLayoutAtom{}));
    };

    template<int M,int K,int Threads,typename T,class ...Tiles>
    struct MMajorConfig{
        using IM = Int<M>;
        using IK = Int<K>;
        // We may tile atom(MxK) in both M and K dims.
        // Ex:(BM,BK,BKStages,BMTiles)
        using TileShape = decltype(flatten(prepend(Tiles{}...,tuple<IM,IK>{})));

        // Layout in the view of modeling
        using LogicalLayoutAtom = Layout<Shape<IM,_8>,Stride<_1,IM>>;


        using SMemLayoutAtom = typename SMemLayoutAtomTrait<M,LogicalLayoutAtom>::SMemLayoutAtom;

        using SMemLayout = decltype(tile_to_shape(SMemLayoutAtom{},TileShape{}));


        using EPT = typename EPTTrait<8,K*M,Threads>::EPT;
        // Threads along M 
        using TM = Int<int(M / EPT{})>;
        // Threads along K
        using TK = Int<int(Threads / TM{})>;

        using VecType = cutlass::AlignedArray<T,EPT{}>;

        using ThrLayout = Layout<Shape<TM,TK>,Stride<_1,TM>>;
        using ValLayout = Layout<Shape<EPT,_1>>;
        
        using AsyncCopyMeta = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<VecType>,T>{},
                                                       ThrLayout{},
                                                       ValLayout{}));
        
        using Copy          = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<VecType>,T>{},
                                                       ThrLayout{},
                                                       ValLayout{}));

        // cp.async may be invalid when cp_size is not 16 bytes 
        using AsyncCopy = std::conditional_t<EPT{}==8,AsyncCopyMeta,Copy>;

    };

    template<int K,int M,int Threads,typename T,class ...Tiles>
    struct KMajorConfig{
        using IM = Int<M>;
        using IK = Int<K>;
        // We may tile atom(MxK) in both M and K dims.
        // Ex:(BM,BK,BKStages,BMTiles)
        using TileShape = decltype(flatten(prepend(Tiles{}...,tuple<IM,IK>{})));

        // Layout in the view of modeling
        using LogicalLayoutAtom = Layout<Shape<_8,IK>,Stride<IK,_1>>;


        using SMemLayoutAtom = typename SMemLayoutAtomTrait<K,LogicalLayoutAtom>::SMemLayoutAtom;

         using SMemLayout = decltype(tile_to_shape(SMemLayoutAtom{},TileShape{}));

        using EPT = typename EPTTrait<8,K*M,Threads>::EPT;
        // Threads along K
        using TK = Int<int(K / EPT{})>;
        // Threads along M
        using TM = Int<int(Threads / TK{})>;

        using VecType = cutlass::AlignedArray<T,EPT{}>;

        using ThrLayout = Layout<Shape<TM,TK>,Stride<TK,_1>>;
        using ValLayout = Layout<Shape<_1,EPT>>;

        using AsyncCopyMeta = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<VecType>,T>{},
                                                       ThrLayout{},
                                                       ValLayout{}));
        
        using Copy          = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<VecType>,T>{},
                                                       ThrLayout{},
                                                       ValLayout{}));

        // cp.async may be invalid when cp_size is not 16 bytes 
        using AsyncCopy = std::conditional_t<EPT{}==8,AsyncCopyMeta,Copy>;

    };
}
}
}
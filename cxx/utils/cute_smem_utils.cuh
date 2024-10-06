#pragma once
#include "cute/tensor.hpp" 
#include <cuda.h>


namespace LAB{

namespace CUDA{

namespace SMem{
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
    // Border of recursion
    template<int TotalEleNum,int Threads>
    struct EPTTrait<0,TotalEleNum,Threads>{
        using EPT = _0;
    };

    
    template<int EleSize,int Cols,typename LogicalLayoutAtom>
    struct SMemLayoutAtomTrait{
        using SMemLayoutAtom = LogicalLayoutAtom;  // Data is naturally interleaved in smem. Ex:Cols=40
    };

    // Swizzle for Half 
    template<typename LogicalLayoutAtom>
    struct SMemLayoutAtomTrait<2,16,LogicalLayoutAtom>{
        using SMemLayoutAtom = decltype(composition(Swizzle<1,3,3>{},LogicalLayoutAtom{}));
    };

    template<typename LogicalLayoutAtom>
    struct SMemLayoutAtomTrait<2,32,LogicalLayoutAtom>{
        using SMemLayoutAtom = decltype(composition(Swizzle<2,3,3>{},LogicalLayoutAtom{}));
    };

    template<typename LogicalLayoutAtom>
    struct SMemLayoutAtomTrait<2,64,LogicalLayoutAtom>{
        using SMemLayoutAtom = decltype(composition(Swizzle<3,3,3>{},LogicalLayoutAtom{}));
    };

    template<typename LogicalLayoutAtom>
    struct SMemLayoutAtomTrait<2,128,LogicalLayoutAtom>{
        using SMemLayoutAtom = decltype(composition(Swizzle<3,3,4>{},LogicalLayoutAtom{}));
    };
    
    // Swizzle for Byte
    template<typename LogicalLayoutAtom>
    struct SMemLayoutAtomTrait<1,32,LogicalLayoutAtom>{
        using SMemLayoutAtom = decltype(composition(Swizzle<1,3,4>{},LogicalLayoutAtom{}));
    };
    
    template<typename LogicalLayoutAtom>
    struct SMemLayoutAtomTrait<1,64,LogicalLayoutAtom>{
        using SMemLayoutAtom = decltype(composition(Swizzle<2,3,4>{},LogicalLayoutAtom{}));
    };

    template<typename LogicalLayoutAtom>
    struct SMemLayoutAtomTrait<1,128,LogicalLayoutAtom>{
        using SMemLayoutAtom = decltype(composition(Swizzle<3,3,4>{},LogicalLayoutAtom{}));
    };

    

    template<int M,int K,int Threads,typename T,class ...Tiles>
    struct MMajorConfig{
        using IM = Int<M>;
        using IK = Int<K>;
     
        static constexpr int EleSize = sizeof(T);
        // Layout in the view of modeling
        using LogicalLayoutAtom = Layout<Shape<IM,_8>,Stride<_1,IM>>;
        using SMemLayoutAtom = typename SMemLayoutAtomTrait<EleSize,M,LogicalLayoutAtom>::SMemLayoutAtom;

        
        // We may tile atom(MxK) in both M and K dims.
        // Ex:(BM,BK,BKStages,BMTiles)
        using TileShape = decltype(flatten(prepend(Tiles{}...,tuple<IM,IK>{})));
        using SMemLayout = decltype(tile_to_shape(SMemLayoutAtom{},TileShape{}));


        static constexpr int MaxEPT = 16 / EleSize;
        using EPT = typename EPTTrait<MaxEPT,M*K,Threads>::EPT;
        static_assert(EPT{} != 0);
        
        // Threads along M 
        using TM = Int<int(M / EPT{})>;
        // Threads along K
        using TK = Int<int(Threads / TM{})>;
        using ThrLayout = Layout<Shape<TM,TK>,Stride<_1,TM>>;
        using ValLayout = Layout<Shape<EPT,_1>>;
        
        using VecType = cutlass::AlignedArray<T,EPT{}>;
       
        
        using Copy          = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<VecType>,T>{},
                                                       ThrLayout{},
                                                       ValLayout{}));
        
        using AsyncCopyMeta = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<VecType>,T>{},
                                                       ThrLayout{},
                                                       ValLayout{}));
        // cp.async may be invalid when cp_size is not 16 bytes 
        using AsyncCopy = std::conditional_t<EPT{}==8,AsyncCopyMeta,Copy>;

    };

    template<int K,int M,int Threads,typename T,class ...Tiles>
    struct KMajorConfig{
        using IM = Int<M>;
        using IK = Int<K>;

        static constexpr int EleSize = sizeof(T);

        // Layout in the view of modeling
        using LogicalLayoutAtom = Layout<Shape<_8,IK>,Stride<IK,_1>>;
        using SMemLayoutAtom = typename SMemLayoutAtomTrait<EleSize,K,LogicalLayoutAtom>::SMemLayoutAtom;
        
        // We may tile atom(MxK) in both M and K dims.
        // Ex:(BM,BK,BKStages,BMTiles)
        using TileShape = decltype(flatten(prepend(Tiles{}...,tuple<IM,IK>{})));
        using SMemLayout = decltype(tile_to_shape(SMemLayoutAtom{},TileShape{}));

        
        static constexpr int MaxEPT = 16 / EleSize;
        using EPT = typename EPTTrait<MaxEPT,M*K,Threads>::EPT;
        static_assert(EPT{} != 0);

        // Threads along K
        using TK = Int<int(K / EPT{})>;
        // Threads along M
        using TM = Int<int(Threads / TK{})>;

        using ThrLayout = Layout<Shape<TM,TK>,Stride<TK,_1>>;
        using ValLayout = Layout<Shape<_1,EPT>>;

        using VecType = cutlass::AlignedArray<T,EPT{}>;

        
        using Copy          = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<VecType>,T>{},
                                                       ThrLayout{},
                                                       ValLayout{}));

        using AsyncCopyMeta = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<VecType>,T>{},
                                                       ThrLayout{},
                                                       ValLayout{}));
        // cp.async may be invalid when cp_size is not 16 bytes 
        using AsyncCopy = std::conditional_t<EPT{}==MaxEPT,AsyncCopyMeta,Copy>;

    };
}
}
}
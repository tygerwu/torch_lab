#pragma once
#include <math.h>
#include "utils/macro.cuh"

namespace LAB{
namespace CUDA{

namespace SM80{

template<typename RFX,typename RFXT>
__forceinline__ __device__ void TransposeX(const RFX& rfx,RFXT& rfxt){
    // Ex: RFX: (2,2),QKMMA_ValTile_BM,QKMMA_ValTile_BN
    // Row2Major -> RowMajor
    constexpr int MMATile_N=size<2>(RFX{});
    constexpr int MMATile_M=size<1>(RFX{}); 

    constexpr int Cols=2 * MMATile_N;
    constexpr int Rows=2 * MMATile_M; 

    const float* rfx_ptr=rfx.data();
    float* rfxt_ptr=rfxt.data();

    CUTE_UNROLL
    for(int i=0; i<Rows; i++){
        CUTE_UNROLL
        for(int j=0; j<Cols/2; j++){
            CUTE_UNROLL
            for(int d=0; d<2; d++){
                rfxt_ptr[i*Cols + j*2 + d]=rfx_ptr[i*2 + j*2*Rows + d];
            }
        }
    }

}



template<typename T,typename RFXT,typename PrevSum,typename PrevMax,typename RP,typename RFO>
__forceinline__ __device__ void Update(RFXT& rfxt,PrevSum& r_prev_sum,PrevMax& r_prev_max,RP& rp,RFO& rfo,
                                       float log2_scale){

    // Ex:
    //   RO: (2,2),QKMMA_ValTile_BM,QKMMA_ValTile_BN2,BN2Num
    //   RP: (2,2,2),PVMMA_ValTile_BM,PVMMA_ValTile_BN
    constexpr int O_MMATile_BM =size<1>(shape(RFO{}));
    constexpr int O_C2s        =size<2>(shape(RFO{})) * size<3>(shape(RFO{}));
    constexpr int P_MMATile_BN =size<2>(shape(RP{}));
    constexpr int Rows  =O_MMATile_BM * 2;
    constexpr int Cols_P=P_MMATile_BN * 2 * 2;

    float* rfxt_ptr=rfxt.data();
    CUTE_UNROLL
    for(int i=0; i<Rows; i++){
        float* rfxt_row_ptr= rfxt_ptr + i * Cols_P;
        // Max
        float cur_max=-INFINITY;
        CUTE_UNROLL
        for(int j=0; j<Cols_P; j++){
            auto x=rfxt_row_ptr[j];
            cur_max=max(cur_max, x);
        }
        // Reduce across threads
        CUTE_UNROLL 
        for(int d=2; d>=1; d=d>>1){
            cur_max=max(cur_max,__shfl_xor_sync(FULL_MASK,cur_max,d,4));
        }
        float prev_max=r_prev_max(i);
        float gmax=max(cur_max,prev_max); 


        float cur_sum=0;
        CUTE_UNROLL
        for(int j=0; j<Cols_P; j++){
            auto x=rfxt_row_ptr[j];
            auto exp_x=exp2f((x - gmax) * log2_scale);
            rfxt_row_ptr[j]=exp_x;
            cur_sum += exp_x;
        }
        // Reduce across threads
        CUTE_UNROLL
        for(int d=2; d>=1; d=d>>1){
            cur_sum += __shfl_xor_sync(FULL_MASK,cur_sum,d,4);
        }

        float prev_sum=r_prev_sum(i);
        float prev_exp=exp2f((prev_max - gmax) * log2_scale);

        float gsum=prev_sum * prev_exp + cur_sum;
        float o_scale=prev_exp;

        // Update params 
        r_prev_sum(i)=gsum;
        r_prev_max(i)=gmax;

        int atom_start =(i / 2) * 8;
        // (i % 2) * 2
        int off_in_atom=((i & 1) << 1); 

        // Update X
        CUTE_UNROLL
        for(int j=0; j<P_MMATile_BN; j++){
            auto rp_ptr= rp.data() + atom_start + off_in_atom + j*Rows*4;
            rp_ptr[0] = static_cast<T>(rfxt_row_ptr[j*4]);
            rp_ptr[1] = static_cast<T>(rfxt_row_ptr[j*4+1]);
            rp_ptr[4] = static_cast<T>(rfxt_row_ptr[j*4+2]);
            rp_ptr[5] = static_cast<T>(rfxt_row_ptr[j*4+3]);

            //(reinterpret_cast<__half2*>(rp_ptr))[0]  =__float22half2_rn((reinterpret_cast<const float2*>(rfxt_row_ptr+j*4))[0]);
            //(reinterpret_cast<__half2*>(rp_ptr+4))[0]=__float22half2_rn((reinterpret_cast<const float2*>(rfxt_row_ptr+j*4+2))[0]);
        }

        // Update O
        CUTE_UNROLL
        for(int j=0; j<O_C2s; j++){
            auto rfo_ptr=rfo.data() + i*2 + j*2*Rows;
            for(int d=0; d<2; d++){
                rfo_ptr[d] *= o_scale;
            }
        }
    }
}

template<typename T,typename RFO,typename RHO,typename PrevSum>
__forceinline__ __device__ void RescaleO(const RFO& rfo,RHO& rho,PrevSum& r_prev_sum){
    // Ex:
    //   RO: (2,2),QKMMA_ValTile_BM,QKMMA_ValTile_BN,BN2Num. Row2Major
    constexpr int Rows=size<1>(RFO{}) * 2;
    constexpr int C2s =size<2>(RFO{}) * size<3>(RFO{});

    const float* rfo_ptr=rfo.data();
    T* rho_ptr=rho.data();

    CUTE_UNROLL
    for(int i=0; i<Rows; ++i){

        float r_gsum=r_prev_sum(i);

        const float* rfo_row_ptr=rfo_ptr + i*2;
        T* rho_row_ptr=rho_ptr + i*2;
        
        CUTE_UNROLL
        for(int j=0; j<C2s; ++j){
            CUTE_UNROLL
            for(int d=0; d<2; d++){
                int offset=j*Rows*2 + d;
                rho_row_ptr[offset]=static_cast<T>(rfo_row_ptr[offset] / r_gsum);
            }
        }
    }
}


}
}
}
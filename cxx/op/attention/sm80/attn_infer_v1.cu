#include "attn_infer_v1_config.cuh"
#include "op/attention/params.cuh"
#include "op/attention/api.cuh"
#include "attn_utils.cuh"

namespace LAB{
namespace CUDA{

namespace SM80{


template<typename T,int HDSize>
__global__ void AttentionInferV1Kernel(AttnInferParams params){

    using HD = Int<HDSize>;
    using CFG = typename AttentionInferV1ConfigTratis<T,HDSize>::CFG;                      
    using BM  = typename CFG::BM;
    using BN  = typename CFG::BN;
    using BK  = typename CFG::BK;
    using BK2 = typename CFG::BK2;
    using BN2 = typename CFG::BN2;

    using BKNum    = typename CFG::BKNum;
    using BK2Num   = typename CFG::BK2Num;
    using BN2Num   = typename CFG::BN2Num;
    using BKTiles  = typename CFG::BKTiles;
    using BK2Tiles = typename CFG::BK2Tiles;


    extern __shared__ char smem[];

    // Positions
    int tid = threadIdx.x;
    int bm_id    = blockIdx.x;        // qo_seqlen
    int head_id  = blockIdx.y;        // head_num
    int batch_id = blockIdx.z;        // batch

    // Instances
    auto tiled_g2s_q    = typename CFG::G2SCopyQ{};
    auto tiled_g2s_k    = typename CFG::G2SCopyK{};
    auto tiled_g2s_v    = typename CFG::G2SCopyV{};
    auto tiled_qk_mma   = typename CFG::QKMMA{};
    auto tiled_pv_mma   = typename CFG::PVMMA{};

    auto tiled_s2r_q    = typename CFG::S2RCopyQ{};
    auto tiled_s2r_k    = typename CFG::S2RCopyK{};
    auto tiled_s2r_v    = typename CFG::S2RCopyV{};

    auto tiled_r2s_o    = typename CFG::R2SCopyO{};
    auto tiled_s2g_o    = typename CFG::S2GCopyO{};

    auto g2s_q = tiled_g2s_q.get_slice(tid);
    auto g2s_k = tiled_g2s_k.get_slice(tid);
    auto g2s_v = tiled_g2s_v.get_slice(tid);

    auto s2r_q = tiled_s2r_q.get_slice(tid);
    auto s2r_k = tiled_s2r_k.get_slice(tid);
    auto s2r_v = tiled_s2r_v.get_slice(tid);

    auto r2s_o = tiled_r2s_o.get_slice(tid);
    auto s2g_o = tiled_s2g_o.get_slice(tid);

    int head_num  = params.head_num;
    int batch     = params.batch;
    int kv_seqlen = params.kv_seqlen;
    int qo_seqlen = params.qo_seqlen;

    int qo_batch_stride = params.q_batch_stride;
    int qo_head_stride  = params.q_head_stride;
    int qo_seq_stride   = params.q_seq_stride;
    
    int kv_batch_stride = params.k_batch_stride;
    int kv_head_stride  = params.k_head_stride;
    int kv_seq_stride   = params.k_seq_stride;

    float log2_scale    = params.log2_scale;


    int bn_num = UP_DIV(kv_seqlen,BN{});

    // GMem Tensors
    auto gqo_layout = make_layout(make_shape(qo_seqlen,HD{},head_num,batch),
                                  make_stride(qo_seq_stride,_1{},qo_head_stride,qo_batch_stride));

    auto gk_layout  = make_layout(make_shape(kv_seqlen,HD{},head_num,batch),
                                  make_stride(kv_seq_stride,_1{},kv_head_stride,kv_batch_stride));
    
    auto gv_layout  = make_layout(make_shape(HD{},kv_seqlen,head_num,batch),
                                  make_stride(_1{},kv_seq_stride,kv_head_stride,kv_batch_stride));

    auto gq = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(params.q_ptr)),gqo_layout);
    auto gk = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(params.k_ptr)),gk_layout);
    auto gv = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(params.v_ptr)),gv_layout);
    auto go = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.o_ptr)),gqo_layout);

    
    // GMem Block 
    auto gq_block = local_tile(gq,make_shape(BM{},BK{}),make_coord(bm_id,_,head_id,batch_id)); // (BM,BK,BKNum)
    auto go_block = local_tile(go,make_shape(BN{},BK{}),make_coord(bm_id,_,head_id,batch_id)); // ~
    auto gk_head  = local_tile(gk,make_shape(BN{},BK{}),make_coord(_,_,head_id,batch_id));     // BN,BK,BNNum,BKNum
    auto gv_head  = local_tile(gv,make_shape(BN2{},BK2{}),make_coord(_,_,head_id,batch_id));   // BN2,BK2,BN2Num_HD,BKNum_KVSEQ


    // SMem Tensors
    auto sq = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem + typename CFG::SOffQ{})),typename CFG::SMemLayoutQ{});
    auto sk = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem + typename CFG::SOffK{})),typename CFG::SMemLayoutK{});
    auto sv = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem + typename CFG::SOffV{})),typename CFG::SMemLayoutV{});
    auto so = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem + typename CFG::SOffO{})),typename CFG::SMemLayoutO{});


    // Load Q into SMem as early as possible
    auto g2s_src_q = g2s_q.partition_S(gq_block);   //(8,1),G2S_ValTile_BM,G2S_ValTile_BK,BKNum
    auto g2s_dst_q = g2s_q.partition_D(sq);         // ~
  
    copy(tiled_g2s_q,g2s_src_q,g2s_dst_q);
    cp_async_fence();

    // G2S K,V
    auto g2s_src_k = g2s_k.partition_S(gk_head);     // (8,1),G2S_ValTile_BN,G2S_ValTile_BK,bn_num,BKNum
    auto g2s_dst_k = g2s_k.partition_D(sk);          // (8,1),G2S_ValTile_BN,G2S_ValTile_BK,BKNum

    auto g2s_src_v = g2s_v.partition_S(gv_head);     // (8,1),G2S_ValTile_BN2,G2S_ValTile_BK2,BN2Num,BK2Num_KVSEQ
    auto g2s_dst_v = g2s_v.partition_D(sv);          // (8,1),G2S_ValTile_BN2,G2S_ValTile_BK2,BN2Num,BK2Num_BN

    auto IssueG2SK = [&](int bn){
        auto g2s_src_k_view = g2s_src_k(_,_,_,bn,_); 
        for_each(make_int_sequence<BKNum{}>{},[&](auto bk){
            // Do pileines along BK
            copy(tiled_g2s_k,g2s_src_k_view(_,_,_,bk),g2s_dst_k(_,_,_,bk));
            cp_async_fence();
        });
    };

    auto IssueG2SV = [&](int bn){
        int off = bn * BK2Num{};
        for_each(make_int_sequence<BK2Num{}>{},[&](auto bk2){
            // Do pileines along BK2
            copy(tiled_g2s_v,g2s_src_v(_,_,_,_,off + bk2),g2s_dst_v(_,_,_,_,bk2));
            cp_async_fence();
        });
    };


    // Reg Q,K,V
    auto rq = make_tensor<T>(typename CFG::RShapeQ{});   // (2,2,2),QKMMA_ValTile_BM,2
    auto rk = make_tensor<T>(typename CFG::RShapeK{});   // (2,2),  QKMMA_ValTile_BN,2
    auto rv = make_tensor<T>(typename CFG::RShapeV{});
    auto rp = make_tensor<T>(typename CFG::RShapeP{});

    // S2R Q,K,V
    auto s2r_src_q_bks  = s2r_q.partition_S(sq);         // (8,1),S2R_ValTile_BM,S2R_ValTile_BK,BKNum
    auto s2r_dst_q      = s2r_q.retile_D(rq);            // (8,1),S2R_ValTile_BM,2

    auto s2r_src_k_bks  = s2r_k.partition_S(sk);         // (8,1),S2R_ValTile_BN,S2R_ValTile_BK,BKNum
    auto s2r_dst_k      = s2r_k.retile_D(rk);            // (8,1),S2R_ValTile_BN,2

    auto s2r_src_v_bn2bk2s = s2r_v.partition_S(sv);      // (8,1),(S2R_ValTile_BN2),S2R_ValTile_BK2,BN2Num,BK2Num
    auto s2r_dst_v         = s2r_v.retile_D(rv);         // (8,1),(S2R_ValTile_BN2),2


    // Reg HO
    auto rho = make_tensor<T>(typename CFG::RShapeO{});
    // R2S
    //  s2r_tile_mn : (PVMMA_M,PVMMA_N), ex:64x16
    auto r2s_src_ho = group_diff<1,0>(flatten(r2s_o.retile_S(rho)));   // ((2),S2RAtom_ValTile_PVMMA_M,S2RAtom_ValTile_PVMMA_N,S2R_ValTile_BM,S2R_ValTile_BN2,BN2Num
    auto r2s_dst_ho = group_diff<1,0>(flatten(r2s_o.partition_D(so))); // ~

    // if(thread0()){
    //     Print("r2s_src_ho:",r2s_src_ho);
    //     Print("r2s_dst_ho:",r2s_dst_ho);
    // }

    // S2G
    auto s2g_src_ho = s2g_o.partition_S(so);            // (8,1),S2G_ValeTile_BM,S2G_ValeTile_BN2,BN2Num
    auto s2g_dst_ho = s2g_o.partition_D(go_block);      // ~


    // Reg accumulators
    auto rfx = make_tensor<float>(typename CFG::RShapeX{});     //(2,2), QKMMA_ValTile_BM,QKMMA_ValTile_BN
    auto rfo = make_tensor<float>(typename CFG::RShapeO{});     //(2,2), PVMMA_ValTile_BM,PVMMA_ValTile_BN2,BN2Num

    // QKMMA
    auto QK_MMA = [&](int bk){
        // Double buffer
        int ld_id = 0;
        int st_id = 0;

        auto s2r_src_q = s2r_src_q_bks(_,_,_,bk);
        auto s2r_src_k = s2r_src_k_bks(_,_,_,bk);

        // Prefetch for 1st mma
        copy(tiled_s2r_q,s2r_src_q(_,_,0),s2r_dst_q(_,_,st_id));
        copy(tiled_s2r_k,s2r_src_k(_,_,0),s2r_dst_k(_,_,st_id));
        st_id ^= 1;
        for(int i=0; i<BKTiles{}; i++){
            if(i+1<BKTiles{}){
                // Prefetch for next round
                copy(tiled_s2r_q,s2r_src_q(_,_,i+1),s2r_dst_q(_,_,st_id));
                copy(tiled_s2r_k,s2r_src_k(_,_,i+1),s2r_dst_k(_,_,st_id));
                st_id ^= 1;
            }

            gemm(tiled_qk_mma,rfx,rq(_,_,ld_id),rk(_,_,ld_id),rfx);
            ld_id ^= 1;
        }
    };

    // if(thread0()){
    //     Print("rp:",rp);
    // }

    auto PV_MMA = [&](int bk2){
        for(int j=0; j<BN2Num{}; j++){
            int st_id = 0;
            int ld_id = 0;

            auto s2r_src_v = s2r_src_v_bn2bk2s(_,_,_,j,bk2);    // (8,1),(S2R_ValTile_BN2),S2R_ValTile_BK2
            copy(tiled_s2r_v,s2r_src_v(_,_,0),s2r_dst_v(_,_,st_id));
            st_id ^= 1;

            for(int i=0; i<BK2Tiles{}; i++){
                if(i+1<BK2Tiles{}){
                    copy(tiled_s2r_v,s2r_src_v(_,_,i),s2r_dst_v(_,_,st_id));
                    st_id ^= 1;
                }

                gemm(tiled_pv_mma,rfo(_,_,_,j),rp(_,_,bk2*BK2Tiles{}+i),rv(_,_,ld_id),rfo(_,_,_,j));
                ld_id ^= 1;
            }
        }   
    };

    
    // Reg softmax params
    auto r_prev_sum = make_tensor<float>(typename CFG::RShapeSM{});
    auto r_prev_max = make_tensor<float>(typename CFG::RShapeSM{});
    fill(r_prev_sum,0); 
    fill(r_prev_max,-INFINITY); 

    // Wait for SMem Q
    cp_async_wait<0>();

    // Prefetch
    IssueG2SK(0);
    for(int bn=0; bn<bn_num; bn++){
        clear(rfx);
        IssueG2SV(bn);
        // QK Gemm
        for_each(make_index_sequence<BKNum{}>{},[&](auto i){
            // Wait for K
            cp_async_wait<BKNum{} + BK2Num{} - i - 1>();
            __syncthreads();
            QK_MMA(i);
        });
        // Prefetch K for next round
        if(bn+1<bn_num){
            IssueG2SK(bn+1);
        }

        auto rfxt = make_tensor<float>(typename CFG::RSizeX{});
        TransposeX(rfx,rfxt);
 
        Update(rfxt,r_prev_sum,r_prev_max,rp,rfo,log2_scale);

        // PV Gemm
        for_each(make_index_sequence<BK2Num{}>{},[&](auto i){
            // Wait for V
            cp_async_wait<BKNum{} + BK2Num{} - i - 1>();
            __syncthreads();
            PV_MMA(i);
        });
    }

    RescaleO<T>(rfo,rho,r_prev_sum);

    // r2s
    copy(tiled_r2s_o,r2s_src_ho,r2s_dst_ho);
    __syncthreads();
    // s2g 
    copy(tiled_s2g_o,s2g_src_ho,s2g_dst_ho);

    // if(thread0()){
    //     Print("r2s_src_ho:",r2s_src_ho);
    //     Print("r2s_dst_ho:",r2s_dst_ho);

    //     Print("s2g_src_ho:",s2g_src_ho);
    //     Print("s2g_dst_ho:",s2g_dst_ho);
    // }
}


template<typename T,int HD>
static void Launch(const AttnInferParams& params,cudaStream_t stream){
    using CFG = typename AttentionInferV1ConfigTratis<T,HD>::CFG;

#ifndef __CUDA_ARCH__
    CFG{}.print();
#endif

    auto func = AttentionInferV1Kernel<T,HD>;
    int BM = typename CFG::BM{};
    int threads = typename CFG::Threads{};
    int smem_bytes = CFG::SBytes;

    int qo_seqlen = params.qo_seqlen;
    int head_num  = params.head_num;
    int batch     = params.batch;

    if(smem_bytes >= (48 << 10)){
        CUDA_ERROR_CHECK(cudaFuncSetAttribute(func,cudaFuncAttributeMaxDynamicSharedMemorySize,smem_bytes));
    }
    dim3 grid(UP_DIV(qo_seqlen,BM),head_num,batch);
    dim3 block(threads);

    func<<<grid,block,smem_bytes,stream>>>(params);

}


template<>
void AttentionInferV1<half_t>(const AttnInferParams& params,cudaStream_t stream){
    
    int head_dim = params.head_dim;
    if(head_dim == 64){
        Launch<half_t,64>(params,stream);
    }else if(head_dim == 128){
        Launch<half_t,128>(params,stream);
    }
}

}
}

}
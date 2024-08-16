
import torch

# FlashAttention wrriten in python

def attn_fwd_head(q,k,v,softmax_scale,casual):
    # q,o:[cur_qo_seqlen,HD]
    # k,v:[cur_kv_seqlen,HD]
    assert q.size() == 2 
    
    cur_qo_seqlen = q.size()[0]
    cur_kv_seqlen = v.size()[0]
    
    BM = 128 
    BN = 128 
    for bm in range(0,cur_qo_seqlen,BM):
        for bn in range(0,cur_kv_seqlen,BN):
            pass 

def attn_fwd_head_mm(q,k,v,softmax_scale,casual):
    io_dtype = q.dtype
    # q,o:[cur_qo_seqlen,HD]
    # k,v:[cur_kv_seqlen,HD]
    kt = k.transpose(-2,-1)
    x = (q @ kt).to(torch.float32)
    
    p = torch.softmax(x * softmax_scale,dim=-1).to(io_dtype)
    
    o = p @ v 
    
    return o  

        
def attn_fwd(q,k,v,softmax_scale,casual=False,use_mm=True):
    # q,o: [B,QO_SEQ_LEN,HN,HD]
    # k,v: [B,KV_SEQ_LEN,HN,HD]
    
    o = torch.empty_like(q) 
    
    batch = q.size()[0]
    head_num = q.size()[2]
    
    head_func = attn_fwd_head_mm if use_mm else attn_fwd_head
    
    for b in range(batch):
        for h in range(head_num):
            o[b,:,h,:] = head_func(q[b,:,h,:],k[b,:,h,:],v[b,:,h,:],softmax_scale,casual)
    
    return o  
    


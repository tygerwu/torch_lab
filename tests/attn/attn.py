
import torch

# FlashAttention wrriten in python

def attn_fwd_head(q,k,v,softmax_scale,casual):
    # q,o:[cur_qo_seqlen,HD]
    # k,v:[cur_kv_seqlen,HD]
    assert q.size() == 2 
    
    in_device = q.device
    io_dtype  = q.dtype
    cur_qo_seqlen = q.size()[0]
    cur_kv_seqlen = v.size()[0]
    hd = q.size()[1]
    
    BM = 128 
    BN = 128 
    
    o = torch.empty_like(q)
    for bm in range(0,cur_qo_seqlen,BM):
        # Runtime BM
        RBM = min(BM,cur_qo_seqlen-bm)
        q_block = q[bm:bm+RBM,:]
        
        prev_sum = torch.zeros(RBM,device=in_device,dtype=torch.float32)
        prev_max = torch.empty_like(prev_sum)
        torch.fill(prev_max,-torch.inf)
        
        o_block = torch.zeros((RBM,hd),device=in_device,dtype=torch.float32)
        for bn in range(0,cur_kv_seqlen,BN):
            RBN = min(BN,cur_kv_seqlen-bn)      # for irregular case
            k_block = k[bn:bn+RBN,:]
            v_block = v[bn:bn+RBN,:]
            
            x_block = q_block @ k_block.transpose(-2,-1).to(torch.float32)
            
            # online softmax
            cur_max = x_block.max(-1,keepdim=True)[0]
            g_max = torch.maximum(cur_max,cur_max)
            
            cur_sum = torch.sum(torch.exp((x_block-cur_max.view(-1,1)) * softmax_scale), dim=-1)
            g_sum = prev_sum * torch.exp((prev_max-g_max)*softmax_scale) + cur_sum * torch.exp((cur_max-g_max)*softmax_scale)
            
            # update o
            o_scale = torch.exp((prev_max-g_max)*softmax_scale)
            o_block *= o_scale.view(-1,1)
            
            # update x
            x_block = torch.exp(softmax_scale * (x_block - g_max.view(-1,1))).to(io_dtype)
        
            # update params
            prev_sum = g_sum 
            prev_max = g_max
            
            o_block += torch.matmul(x_block,v_block).to(torch.float32) 
            
        
        # rescale 
        o_block /= prev_sum.view(-1,1)
        o[bm:bm+RBM,:] = o_block.to(io_dtype)
    
    return o

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
    


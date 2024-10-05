from tests.test_utils import(
    load_toch_lab,create_shape_data,check,seed_everything,profile_func
    
)
import torch 
from attn import attn_fwd

load_toch_lab()



def run(batch,hn,hd,qo_seqlen,kv_seqlen,profile,verify, dtype=torch.float16):
    qo_shape = [batch,qo_seqlen,hn,hd]
    kv_shape = [batch,kv_seqlen,hn,hd]
    
    q = torch.from_numpy(create_shape_data(qo_shape,1,2)).to('cuda').to(dtype)
    k = torch.from_numpy(create_shape_data(kv_shape,1,2)).to('cuda').to(dtype)
    v = torch.from_numpy(create_shape_data(kv_shape,1,2)).to('cuda').to(dtype)
    
    
    q = torch.randn(qo_shape,dtype=dtype,device='cuda')
    k = torch.randn(kv_shape,dtype=dtype,device='cuda')
    v = torch.randn(kv_shape,dtype=dtype,device='cuda')
    
    softmax_scale = 2.0
    
    def v1_func():
        return torch.ops.torch_lab.mha_infer(q,k,v,softmax_scale,"attn_infer_v1")

    def v2_func():
        return torch.ops.torch_lab.mha_infer(q,k,v,softmax_scale,"attn_infer_v2")
    
    if profile:
        profile_func(v1_func,2)
        profile_func(v1_func,50,"v1")
    
    if verify:
        torch_out = attn_fwd(q,k,v,softmax_scale)
        lab_out = v2_func()
        print(lab_out)
        print(torch_out)
        check(lab_out,torch_out,l1_thresh=0.01,rtol=1e-2,atol=1e-2)
   
def example():
    batch = 1 
    hn = 2
    hd = 128 
    qo_seqlen = 127
    kv_seqlen = 256
    profile = False
    verify = True
    run(batch,hn,hd,qo_seqlen,kv_seqlen,profile,verify)

seed_everything(0)    
example()

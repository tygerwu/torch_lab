from tests.test_utils import(
    load_toch_lab,create_shape_data,check,seed_everything,profile_func
    
)
import torch 

load_toch_lab()

def torch_varm_gmm(a,b,ms,n,trans_b):
    total_m = sum(ms)
    groups = len(ms) 
    c = torch.empty((total_m,n)).to(a.device).to(a.dtype) 
    
    prev_m = 0 
    for i in range(groups):
        cur_m = ms[i]
        sub_a = a[prev_m:prev_m+cur_m,:]
        sub_b = b[i,:,:]
        
        if trans_b:
            sub_b = sub_b.transpose(-1,-2)
        
        c[prev_m:prev_m+cur_m,:] = sub_a @ sub_b
        prev_m += cur_m
    
    return c 
            
        
    

def run(ms,n,k,trans_b=True,verify=True,profile=False,dtype=torch.float16):
    total_m = sum(ms)
    groups = len(ms) 
    
    b_shape = [groups,n,k] if trans_b else [groups,k,n] 
    a_shape = [total_m,k]
    a = torch.from_numpy(create_shape_data(a_shape,1,2)).cuda().to(dtype)
    b = torch.from_numpy(create_shape_data(b_shape,1,2)).cuda().to(dtype)
    
    a = torch.randn(a_shape,device='cuda',dtype=dtype)
    b = torch.randn(b_shape,device='cuda',dtype=dtype) 
    
    cpu_ms = torch.tensor(ms,dtype=torch.int32).cpu() 
    
    def cutlass2x_sm8x_func():
        return torch.ops.torch_lab.gmm_varm(a,b,cpu_ms,trans_b)
    
    if verify:
        torch_out = torch_varm_gmm(a,b,ms,n,trans_b)
        sm80_out = cutlass2x_sm8x_func()
        
        check(torch_out,sm80_out)
    
def test1():
    ms = [127,128,63]
    n = 128 + 64 
    k = 128 
    trans_b = False 
    verify = True 
    profile = False
    run(ms,n,k=k,trans_b=trans_b,profile=profile,verify=verify)
    
test1()
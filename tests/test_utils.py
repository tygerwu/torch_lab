import os 
import torch
import numpy as np
import time
import math

def load_toch_lab():
    lib_path = os.environ.get("TORCH_LAB_PATH")
    lib_path = '/media/tyger/linux_ssd/codes/cxx_test/torch_lab/build/lib/torch_lab/libtorch_lab.so'
    try:
        torch.ops.load_library(lib_path)
    except Exception as e:
        print(f"An error occurred while loading the library: {e}")
        
        try: 
            torch.ops.load_library('libtorch_lab.so')
        except Exception as e:
            print("Failed to load libtorch")

def l1_loss_abs(a,b):
    flat_a = a.flatten()
    flat_b = b.flatten()
    return torch.sum(torch.abs(flat_a - flat_b)) / flat_a.numel()

def l1_loss_rel(a,b):
    flat_a = a.flatten()
    flat_b = b.flatten()
    return torch.sum(torch.abs(flat_a - flat_b) / torch.abs(flat_b)) / flat_a.numel()


def np_cmp(a,b,rtol,atol):
    flat_a = a.flatten()
    flat_b = b.flatten()
    assert len(flat_a) == len(flat_b)
    
    for i in range(len(flat_a)):
        if not np.isclose(flat_a[i],flat_b[i],rtol=rtol,atol=atol):
            print("{},{}!={}".format(i,flat_a[i],flat_b[i]))
            return

def check(a,b,l1_thresh=0.001,rtol=1e-3,atol=1e-4):
    l1_loss = l1_loss_abs(a,b)
    print("L1 Loss: ", l1_loss) 
    if l1_loss > l1_thresh or math.isnan(l1_loss) :
        np_cmp(a.to(torch.float32).detach().cpu().numpy(),
               b.to(torch.float32).detach().cpu().numpy(),rtol,atol)
        return False 
    return True
        
    
def create_data(size,beg,end):
    v = beg 
    res = []
    for i in range(size):
        res.append(v)
        v = v + 1
        if v >= end:
            v = beg 
    return np.array(res)

def create_shape_data(shape,beg,end,dtype=np.float32):
    size = np.prod(np.array(shape,dtype=np.int32))
    arr = create_data(size,beg,end)
    arr = arr.astype(dtype)
    return np.reshape(arr,shape)

def profile_func(func,loops,msg=None):
    if loops <= 0:
        return 

    torch.cuda.synchronize()
    beg = time.time()
    for i in range(loops):
        func()
    torch.cuda.synchronize()    
    end = time.time()
    
    ms_time = 1000 * (end - beg) / loops 
    
    if msg is not None: 
        print(msg,' ',ms_time)
    return ms_time

def seed_everything(seed: int):
    import random,os
    import torch 
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONSEED'] = str(seed)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
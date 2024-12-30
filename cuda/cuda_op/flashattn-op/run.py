import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import fav1

# Load the CUDA kernel as a python module
# fav1 = load(name='fav1', sources=['flashattn.cc', 'flashattn-v1.cu'], extra_cuda_cflags=['-O2'])

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 32
nh = 12
seq_length = 64
head_dimension = 128

q = torch.randn(batch_size, nh, seq_length, head_dimension).cuda()
k = torch.randn(batch_size, nh, seq_length, head_dimension).cuda()
v = torch.randn(batch_size, nh, seq_length, head_dimension).cuda()


# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

class FlashAttnV1(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v):
        return fav1.forward(q,k,v)

def fav1_(q,k,v):
    FlashAttnV1.apply(q,k,v)

def countElapsedTime(func,ipt):
    [Q,K,V]  = ipt
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # warmup
    for _ in range(5):
        func(Q,K,V)
    
    start.record()
    for _ in range(5):
        func(Q,K,V)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 3

print(f"manual_attn cost {countElapsedTime(manual_attn,[q,k,v])}ms")
print(f"fav1 cost {countElapsedTime(fav1.forward,[q,k,v])}ms")

# print('=== profiling manual attention ===')

# with torch.autograd.profiler.profile(use_cuda=True) as prof:
#     manual_result = manual_attn(q, k, v)
# print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

# print('=== profiling minimal flash attention === ')

# with torch.autograd.profiler.profile(use_cuda=True) as prof:
#     # minimal_result = fav1_(q,k,v) 
#     minimal_result = fav1.forward(q, k, v)
# print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

# print('attn values sanity check:', torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-01))   # 用于检测两个tensor的相似度 |a-b| < rtol*|b| + atol
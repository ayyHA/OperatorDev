#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <float.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/types.h>
#include <torch/extension.h>

#define CEIL_OPERATION(X,Y) ( (X) + ((Y)-1) ) / (Y)
#define FLOAT4(VAL) ((reinterpret_cast<float4*>(&(VAL)))[0])


#define WARP_SIZE 32
#define THREAD_MASK 0xffffffff


/* ---------------------template function--------------------- */
template<const int kWarpSize=WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val){
    #pragma unroll
    for(int xor_mask = kWarpSize>>1; xor_mask>=1; xor_mask>>=1 ){
        val += __shfl_xor_sync(THREAD_MASK,val,xor_mask);
    }
    return val;
}

template<const int kWarpSize=WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val){
    #pragma unroll
    for(int xor_mask = kWarpSize>>1;xor_mask>=1;xor_mask>>=1){
        val += __shfl_xor_sync(THREAD_MASK,val,xor_mask);
    }
    return val;
}

template<const int kWarpSize=WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val){
    float val_f32 = __half2float(val);
    #pragma unroll
    for(int xor_mask = kWarpSize>>1;xor_mask>=1;xor_mask>>=1){
        val_f32 += __shfl_xor_sync(THREAD_MASK,val_f32,xor_mask);
    }
    return val_f32;
}


template<const int NUM_THREADS=128>
__device__ __forceinline__ float block_reduce_sum_f32(float val){
    constexpr int NUM_WARPS = CEIL_OPERATION(NUM_THREADS,WARP_SIZE);
    __shared__ float smem[NUM_WARPS];
    int tid = threadIdx.x;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    val = warp_reduce_sum<WARP_SIZE>(val);
    if(laneId == 0){
        smem[warpId] = val;
    }
    __syncthreads();
    val = laneId < NUM_WARPS ? smem[laneId] : 0.0f;
    val = warp_reduce_sum<NUM_WAPRS>(val);
    return val;
}

/* ---------------------global function--------------------- */
/*
    grid(N*K / K), block(K)
    N = batch_size * seq_length, K = hidden_dim
    
    row-wise
    y' = ( x - mean(x) ) / ( std(x) + eps )  
    y  = gamma * y' + beta
    mean(x) = sum(x) / K
    1 / std(x) = rsqrtf( sum((x-mean(x)) * (x-mean(x))) / K  + eps )

    layernorm_f32_f32_kernel
    {OP}_{input_fp_size{_{ x4 | x2 }}}_{ouput_fp_size}_kernel 
*/
template<const int NUM_THREADS=256>
__global__ void layernorm_f32_f32_kernel(float* x,float* y,float gamma,float beta,int N,int K){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = bid * K + tid;
    float eps = 1e-5f;
    float value = idx<N*K ? x[idx] : 0.0f;
    __shared__ float s_mean = 0.0f;
    __shared__ float s_std = 0.0f;
    float val = block_reduce_sum<NUM_THREADS>(value);
    if(tid == 0)
        s_mean = val / float(K);
    __syncthreads();
    float variance = (value - s_mean) * (value - s_mean);
    variance = block_reduce_sum<NUM_THREADS>(variance);
    if(tid == 0)
        s_std = rsqrtf(variance / K + eps);
    __syncthreads();
    float _y = ( value - s_mean ) * s_std;
    if(idx<N*K)
        y[idx] = _y * gamma + beta;
}

template<const int NUM_THREADS=256/4>
__global__ void layernorm_f32x4_f32_kernel(const float* x,float* y,const float gamma,const float beta,const int N,const int K){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = (bid*K + tid)*4;
    float eps = 1e-5f;
    float4 value4 = FLOAT4(x[idx]);
    float val = idx < N*K ? value4.x + value4.y + value4.z + value.w : 0.0f;
    
    __shared__ float s_mean = 0.0f;
    __shared__ float s_std = 0.0f;
        
    val = block_reduce_sum<NUM_THREADS>(val);
    if(tid == 0)
        s_mean = val / (float)K;
    __syncthreads();
    float4 _x = value4;
    _x.x = value4.x - s_mean;
    _x.y = value4.y - s_mean;
    _x.z = value4.z - s_mean;
    _x.w = value4.w - s_mean;
    float variance = (_x.x * _x.x) + (_x.y * _x.y) + (_x.z * _x.z) + (_x.w * _x.w);
    variance = block_reduce_sum<NUM_THREADS>(variance);
    if(tid == 0)
        s_std = rsqrtf(variance / K + eps);
    __syncthreads();
    _x.x = _x.x * s_std * gamma + beta;
    _x.y = _x.y * s_std * gamma + beta;
    _x.z = _x.z * s_std * gamma + beta;
    _x.w = _x.w * s_std * gamma + beta;
    if(idx < N*K)
        FLOAT4(y[idx]) = _x;
}

/*
    这是一个需要想象的世界:-D

    tx = threadIdx.x;
    ty = threadIdx.y;
    bx = blockIdx.x;
    by = blockIdx.y;

    tid = ty * blockDim.x + tx;
    bid = by * gridDim.x + bx;

    idx = bid * (blockDim.x * blockDim.y) + tid 
 */

#include <cuda_runtime.h>
#include <cstdio>

#define WARP_SIZE 32
#define THREAD_MASK 0xffffffff
#define ceilOperation(int x,int y) ( ( (x) + (y) - 1 ) / (y) )

/* warp_reduce之sum操作的一个模板,vllm经典写法,以warp为单位进行操作,进来的每一个val都是不同的值,
因为是一个warp里的线程一起进来,针对warp级别的操作 */
template<const int kWarpSize=WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val){
    #pragma unroll
    for(int xor_mask = kWarpSize>>1; xor_mask>=1; xor_mask>>=1){
        val += __shfl_xor_sync(THREAD_MASK,val,xor_mask);
    }
    return val
}

/* block_reduce之sum操作的一个模板,注意,这并非是正儿八经的reduce_sum的模板 */
template<const int NUM_THREADS=128>
__device__ __forceinline__ float block_reduce_sum(float val){
    /* 1024个线程每个block最多,即NUM_WARPS<=32 */
    constexpr int NUM_WARPS = ceilOperation(NUM_THREADS,WARP_SIZE);
    int tx = threadIdx.x;
    int warpId = tx / WARP_SIZE;
    int laneId = tx % WARP_SIZE;
    static __shared__ smem[NUM_WARPS];
    val = warp_reduce_sum<WARP_SIZE>(val);  // 每个warp进行reduce_sum操作
    // 类似于AllGather把每个小元素放到smem,然后把smem里的东西给NUM_WARPS们赋予
    if(laneId==0)
        smem[warpId] = val;
    __syncthreads();
    val = laneId<NUM_WARPS?smem[laneId]:0.0f;
    val = warp_reduce_sum<NUM_WARPS>(val);
    return val;
}

/* 
    block_all_reduce_sum,正儿八经的整一个的归约 
    grid(N/128),block(128)
*/
template<const int NUM_THREADS=128>
__global__ void block_all_reduce_sum(float* ipt,float* opt,int N){
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = ceilOperation(NUM_THREADS,WARP_SIZE);
    float val = idx < N ? ipt[idx] : 0.0f;
    __shared__ float smem[NUM_WARPS];
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    val = warp_reduce_sum<WARP_SIZE>(val);
    if(laneId==0)
        smem[warpId] = val;
    __syncthreads();
    val = laneId<NUM_WARPS ? smem[laneId] : 0.0f;
    if(warpId==0){
        val = warp_reduce_sum<NUM_WARPS>(val);
    }
    if(tid==0){
        atomicAdd(opt,val);
    }
}

/* 
    block_all_reduce_sum_vec4 正儿八经的一个归约,加上了向量化访存 
    grid(N/128),block(128/4)
*/
template<const int NUM_THREADS=128/4>
__global__ void block_all_reduce_sum_vec4(float* ipt,float* opt,int N){
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
    constexpr int NUM_WARPS = ceilOperation(NUM_THREADS,WARP_SIZE);
    float4 val4 = float4[idx];
    float val = idx < N ? val4.x + val.y + val.z + val.w : 0.0f;

    constexpr int NUM_WARPS = ceilOperation(NUM_THREADS,WARP_SIZE);
    __shared__ float smem[NUM_WARPS];

    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    val = warp_reduce_sum<WARP_SIZE>(val);
    if(laneId==0)
        smem[warpId] = val;
    __syncthreads();
    val = laneId < NUM_WARPS ? smem[laneId] : 0.0f;
    if(warpId == 0)   
        val = warp_reduce_sum<NUM_WARPS>(val);
    if(tid == 0)
        atomicAdd(opt,val);
}

/* 
    sgemv k32 
    y = a(MxK) * x(Kx1) 

    grid(M/4) block(32,4)
*/
__global__ void sgemv_k32(float* a,float* x,float* y,int M, int K){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int m = bx*blockDim.y+ty;
    int laneId = tx % WARP_SIZE;

    if(m<M){
        int NUM_WAPRS = ceilOperation(K,WARP_SIZE); // 第一次写这里出错
        float val = 0.0f;
        for(int i=0;i<NUM_WAPRS;i++){
            int index = i * WARP_SIZE + laneId;
            // val += a[m*blockDim.x+index]*x[index];  // deftruth写作val+= a[m*K + index] * x[index]这样更好,因为K是ld(row major);
            val += a[m*K + index] * x[index];
        }
        val = warp_reduce_sum<WARP_SIZE>(val);
        if(tx==0)   // deftruth写作laneId==0,但觉得没啥区别
            y[m] = val;
    }
}

/*
    sgemv k128 + vec4
    y = a(MxK) * x(Kx1)

    grid(M/4),block(32,4)
*/
__global__ void sgemv_k128(float* a,float* x,float* y,int M,int K){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int m  = bx * blockDim.y + ty;
    int laneId = tx % WARP_SIZE;
    if(m<M){
        float val = 0.0f;
        float NUM_WARPS = ceilOperation(ceilOperation(K,WARP_SIZE),4);
        for(int i=0;i<NUM_WAPRS;i++){
            float index = (i*WARP_SIZE + laneId)*4;
            float4 a4 = a[m*K + index];
            float4 x4 = x[index];
            val += a4.x * x4.x + a4.y * x4.y + a4.z * x4.z + a4.w * x4.w;
        }
        val = warp_reduce_sum<WARP_SIZE>(val);
        if(laneId == 0)
            y[m] = val;
    }
}

/*
    sgemv k16
    K_ROWS_PER_WARP = 2;
    NUM_WARPS = NUM_THREADS / WARP_SIZE;
    NUM_ROWS = NUM_WARPS * K_ROWS_PER_WARP
    grid(N/NUM_ROWS),block(32,NUM_WARPS)
*/
template<const int K_ROWS_PER_WARP = 2>
__global__ void sgemv_k16(float* a,float* x,float* y,int M,int K){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int K_WARP_SIZE = WARP_SIZE / K_ROWS_PER_WARP;
    int laneId = tx % WARP_SIZE;
    int _laneId = tx % K_WARP_SIZE;
    int m  = (bx * blockDim.y + ty) * K_ROWS_PER_WARP + laneId / K_WARP_SIZE;
    if(m<M){
        int NUM_WARPS = ceilOperation(K,K_WARP_SIZE);
        float val = 0.0f;
        for(int i=0;i<NUM_WAPRS;i++){
            int index = i*K_WARP_SIZE + _laneId;
            val += a[m*K + index] * x[index]; 
        }
        val = warp_reduce_sum<K_WARP_SIZE>(val);
        if(_laneId == 0)
            y[m] = val;
    }
}
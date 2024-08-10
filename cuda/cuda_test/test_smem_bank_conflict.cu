#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

__global__ void smem_t1(uint32_t* src){
    __shared__ uint32_t smem[128];
    int tid = threadIdx.x;
    for(int i=0;i<4;i++){
        smem[i*32+tid] = tid;
    }
    __syncwarp();
    if(tid<16){
        reinterpret_cast<uint2*>(src)[tid]= reinterpret_cast<uint2*>(smem)[tid];
    }
}

__global__ void smem_t2(uint32_t* src){
    __shared__ uint32_t smem[128];
    int tid = threadIdx.x;
    for(int i=0;i<4;i++){
        smem[i*32+tid] = tid;
    }
    __syncwarp();
    if(tid<15 || tid==16){
        reinterpret_cast<uint2*>(src)[tid] = reinterpret_cast<uint2*>(smem)[tid==16?15:tid];
    }
}

__global__ void smem_t3(uint32_t* src){
    __shared__ uint32_t smem[128];
    int tid = threadIdx.x;
    for(int i=0;i<4;i++){
        smem[i*32+tid] = tid;
    }
    __syncwarp();
    reinterpret_cast<uint2*>(src)[tid] = reinterpret_cast<uint2*>(smem)[tid/2];
}

__global__ void smem_t4(uint32_t *src){
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    for(int i=0;i<4;i++){
        smem[i*32+tid] = tid;
    }
    __syncwarp();
    uint32_t idx = (tid>>4)<<3;
    // printf("0:[tid:%d]:[idx:%d]\n",tid,idx);
    idx = (idx == 0) ? idx+tid>>1 : (idx + (tid&1) + (((tid>>2)-4)<<1));
    // printf("1:[tid:%d]:[idx:%d]\n",tid,idx);
    reinterpret_cast<uint2*>(src)[tid] = reinterpret_cast<uint2*>(smem)[idx];
}

__global__ void smem_t5(uint32_t* src){
    __shared__ uint32_t smem[128];
    int tid = threadIdx.x;
    for(int i=0;i<4;i++){
        smem[i*32 + tid] = tid;
    }
    __syncwarp();
    uint32_t idx = tid>>4==0 ? tid : tid-16;
    reinterpret_cast<uint2*>(src)[tid] = reinterpret_cast<uint2*>(smem)[idx];
}


int main(){
    uint32_t* dA;
    size_t sizedA = sizeof(uint32_t) * 128;
    /*
        相当于:
        bank    0   1  ... 31 
               | | | | | | | |
                -   -   -   - 
               | | | | | | | |
                -   -   -   -
               | | | | | | | |
                -   -   -   - 
               | | | | | | | |
     */
    cudaMalloc(&dA,sizedA);
    dim3 grid(1);
    dim3 block(32);

    smem_t1<<<grid,block>>>(dA);
    smem_t2<<<grid,block>>>(dA);
    smem_t3<<<grid,block>>>(dA);
    smem_t4<<<grid,block>>>(dA);
    smem_t5<<<grid,block>>>(dA);
    cudaDeviceSynchronize();

    cudaFree(dA);
    return 0;
}
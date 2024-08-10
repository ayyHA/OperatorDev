#include <cuda_runtime.h>
#include <stdio.h>

/**
    5.
    unroll last warp:0.019~0.021ms 
    syncwarp:        0.018~0.0202ms
    两者相差不大

    下一步,进行完全展开

    volatile这个很重要,它说明了我们src是一个随时可能变化的值,你每次要去内存取?
    不要编译器的优化,别给我从上一次的寄存器中取值,尽管从代码上看寄存器中的值没有变化
 */

const int N = 65536;
const int BLOCKSIZE = 256;
const int nRepeats = 10;

__device__ void last_warp(volatile float* src,int tx){
    src[tx] = src[tx] + src[tx+32];
    src[tx] = src[tx] + src[tx+16];
    src[tx] = src[tx] + src[tx+8];
    src[tx] = src[tx] + src[tx+4];
    src[tx] = src[tx] + src[tx+2];
    src[tx] = src[tx] + src[tx+1];
}

__global__ void reduceV2(float* src,float* dst,int n){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = tx + bx * blockDim.x*2;
    extern __shared__ float smemTmp[];
    smemTmp[tx] = (tid+blockDim.x) < n?src[tid]+src[tid+blockDim.x]:0.0f; 
    __syncthreads();

    for(int s=(blockDim.x >> 1);s>=32;s>>=1){
        if(tx < s){
            smemTmp[tx] = smemTmp[tx] + smemTmp[tx+s];
        }
        __syncthreads();       
    }

    for(int s=16;s>0;s>>=1){
        if(tx < s){
            smemTmp[tx] = smemTmp[tx] + smemTmp[tx+s];
        }
        __syncwarp();       
    }

    // if(tx<32)
    //     last_warp(smemTmp,tx);

    if(tx==0)
        dst[bx] = smemTmp[0];
    // if(tx == 0)  // 用来校验算法是否错误的
    //     atomicAdd(dst,smemTmp[0]);
}

void testPerfomance(float* dSrc){
    dim3 block(BLOCKSIZE);
    dim3 grid(((N+BLOCKSIZE-1)/BLOCKSIZE)>>1);
    
    float allTime=.0f,avgTime=.0f;
    
    size_t sizeDst = sizeof(float) * grid.x;
    float* dDst,*hDst;

    hDst = (float*)malloc(sizeDst);
    cudaMalloc(&dDst,sizeDst);

    for(int i=0;i<nRepeats;i++){
        cudaEvent_t start,end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaMemset(dDst,0,sizeDst);
        memset(hDst,0,sizeDst);

        cudaEventRecord(start);
       
        reduceV2<<<grid,block,sizeof(float)*BLOCKSIZE>>>(dSrc,dDst,N);
        cudaMemcpy(hDst,dDst,sizeDst,cudaMemcpyDeviceToHost);

        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float elpased_time;
        cudaEventElapsedTime(&elpased_time,start,end);
        printf("hDst[%d]: %.6f\n",i,hDst[0]);
        printf("ElapsedTime[%d]: %.6fms\n",i,elpased_time);
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        allTime += elpased_time;
    }
    
    avgTime = allTime / nRepeats;
    printf("AllTime: %.6fms, AvgTime: %.6fms\n",allTime,avgTime);
    
    free(hDst);
    cudaFree(dDst);
}

int main(){
    size_t sizeSrc = sizeof(float) * N;

    float* hSrc;
    float* dSrc;
    hSrc = (float*)malloc(sizeSrc);

    for(int i=0;i<N;i++)
        hSrc[i] = 1.25;

    cudaMalloc(&dSrc,sizeSrc);
    
    cudaMemcpy(dSrc,hSrc,sizeSrc,cudaMemcpyHostToDevice);

    testPerfomance(dSrc);        

    free(hSrc);
    cudaFree(dSrc);
    return 0;
}
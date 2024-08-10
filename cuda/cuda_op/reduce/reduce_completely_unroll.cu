#include <cuda_runtime.h>
#include <stdio.h>

/**
    6. 完全展开,根据输入的blocksize(要是const,不知道为啥constexpr不行)进行判断
       然后就把原本的那个循环进行展开,需要注意这里最大的blocksize>=512而不是1024,
       因为上面的把它一倍长的add了(idle thread那里的工作)
       所以只需要选择范围,考虑范围内的顺序地址和

       需要注意,这里用模板参数,是因为后面编译器可以给我们做优化,因为blocksize实际上是确定的,
       在生成相应的cubin(存疑)时,可以把你的不需要的部分剔除掉,比如下面blocksize>=512那个分支会被剔除掉
 */

const int N = 65536;
const int BLOCKSIZE = 256;
const int nRepeats = 10;


template<unsigned int blocksize>
__device__ void last_warp(volatile float* src,int tx){
    if(blocksize>=64) src[tx] = src[tx] + src[tx+32];
    if(blocksize>=32) src[tx] = src[tx] + src[tx+16];
    if(blocksize>=16) src[tx] = src[tx] + src[tx+8];
    if(blocksize>=8) src[tx] = src[tx] + src[tx+4];
    if(blocksize>=4) src[tx] = src[tx] + src[tx+2];
    if(blocksize>=2) src[tx] = src[tx] + src[tx+1];
}

template<unsigned int blocksize>
__global__ void reduceV2(float* src,float* dst,int n){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = tx + bx * blockDim.x*2;
    extern __shared__ float smemTmp[];
    smemTmp[tx] = (tid+blockDim.x) < n?src[tid]+src[tid+blockDim.x]:0.0f; 
    __syncthreads();

    // 128
    // for(int s=(blockDim.x >> 1);s>32;s>>=1){
    //     if(tx < s){
    //         smemTmp[tx] = smemTmp[tx] + smemTmp[tx+s];
    //     }
    //     __syncthreads();       
    // }
    if(blocksize>=512){
        if(tx< 256)
            smemTmp[tx] = smemTmp[tx] + smemTmp[tx +256];
        __syncthreads();
    }
    if(blocksize>=256){
        if(tx < 128)
            smemTmp[tx] = smemTmp[tx] + smemTmp[tx+128];
        __syncthreads();
    }

    if(blocksize>=128){
        if(tx < 64)
            smemTmp[tx] = smemTmp[tx] + smemTmp[tx+64];
        __syncthreads();
    }

    if(tx<32)
        last_warp<blocksize>(smemTmp,tx);

    // if(tx==0)
    //     dst[bx] = smemTmp[0];
    if(tx == 0)  // 用来校验算法是否错误的
        atomicAdd(dst,smemTmp[0]);
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
       
        reduceV2<BLOCKSIZE><<<grid,block,sizeof(float)*BLOCKSIZE>>>(dSrc,dDst,N);
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
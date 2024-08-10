#include <cuda_runtime.h>
#include <stdio.h>

/**
    2. 第一种写法存在线程束分化问题
    这里通过idx = s*2*tx来部分解决这个问题:
    256个线程=8个warp
    第1次循环:[0,3]warp进入,[4,7]warp不进入
    第2次循环:[0,1]warp进入,[2,7]warp不进入
    第3次循环:[0]warp进入,[1,7]warp不进入
    第4次循环:0号warp前16个线程进入,后16个线程不进入,依旧线程束分化,解决了部分

    与之而来(从0号warp讨论):
    第一次循环:0号线程取0,1; 1号线程取2,3 ; ... ; 16号线程取32,33;其中0%32 == 32%32 [注:共享内存是4B为粒度访问,32个bank]
    此刻发生了smem的bank conflict;而且是2 way bank conflict
    第二次循环:0号线程取0,2; 1号线程取4,6 ; ... ; 8号线程取32,34 ; ... ; 16号线程取64,66 ; ... ; 24号线程取96,98 ...
    此刻发生了smem的bank conflict;而且是4 way bank conflict
 */

const int N = 65536;
const int BLOCKSIZE = 256;
const int nRepeats = 10;

__global__ void reduceV2(float* src,float* dst,int n){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = tx + bx * blockDim.x;
    extern __shared__ float smemTmp[];
    smemTmp[tx] = tid < n?src[tid]:0.0f; 
    __syncthreads();

    for(int s=1;s<blockDim.x;s*=2){
        int idx = 2*s*tx;
        if(idx < blockDim.x){
            smemTmp[idx] = smemTmp[idx] + smemTmp[idx+s];
        }
        __syncthreads();       
    }

    if(tx==0)
        dst[bx] = smemTmp[0];
    // if(tx == 0)  // 用来校验算法是否错误的
    //     atomicAdd(dst,smemTmp[0]);
}

void testPerfomance(float* dSrc){
    dim3 block(BLOCKSIZE);
    dim3 grid((N+BLOCKSIZE-1)/BLOCKSIZE);
    
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
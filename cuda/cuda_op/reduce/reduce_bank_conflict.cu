#include <cuda_runtime.h>
#include <stdio.h>

/**
    3. 第2种写法中解决了线程束分化的问题,但引入了bank conflict(一个warp的不同线程访问到同一个bank的不同地址)
    之所以会出现bank conflict的根本原因,在于我们用smem遍历时,每个线程的读数方式是交错的
    即把32个bank和32个线程对应上,它并不是同一线程沿着列对应读,而是同一线程沿着行方向读,这必然会导致冲突
    因此需要把线程读数的方式进行修改,沿着列读,并且是对应着读,如0号线程读bank0,1号读bank1这样顺序的读数

    通过以下修改,解决了bank conflict的问题
    但随之而来的是:
    第1次迭代:[0,127]线程读对应的数据 只用了1/2的线程没用上
    第2次迭代:[0,63]线程做处理 只用了1/4的线程 
    第3次迭代:[0,31]线程做处理 只用了1/8的线程
    ...
    log2(256) = 8,需要8次迭代,1/2 -> 1/4 -> ... -> 1/256,平均起来不到1/8的线程利用率
    如何解决idle thread
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

    for(int s=(blockDim.x >> 1);s>0;s>>=1){
        if(tx < s){
            smemTmp[tx] = smemTmp[tx] + smemTmp[tx+s];
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
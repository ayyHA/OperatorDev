#include <cuda_runtime.h>
#include <stdio.h>

/**
    4. 针对空闲的线程,我们所能做的,就是让它干多一点活
    比如说原先256个线程处理256个数据,现在256个线程处理512个数据,相应的block数目就会减少一半
    而256个线程处理512个数据,就是让[128,255]线程一起动起来,除了加载数据,也做一次加法

    之后可以看到在迭代时,其实第3次迭代的时候,就只有一个warp在干活了,但是这个时候使用的还是syncthreads,比较浪费
    因此下一个实验通过展开最后一个warp和采用syncwarp做对比
 */

const int N = 65536;
const int BLOCKSIZE = 256;
const int nRepeats = 10;

__global__ void reduceV2(float* src,float* dst,int n){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = tx + bx * blockDim.x*2;
    extern __shared__ float smemTmp[];
    // 要是tid<n但是tid+blockDim.x>n你这个就有问题了
    smemTmp[tx] = (tid+blockDim.x) < n?src[tid]+src[tid+blockDim.x]:0.0f; 
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
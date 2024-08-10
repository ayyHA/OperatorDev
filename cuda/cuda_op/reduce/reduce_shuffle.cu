#include <cuda_runtime.h>
#include <stdio.h>

/**
    7. 寻找合理的block数目,采用grid stride来进行数据的加载,多多复用线程,别1个线程就瞅着那点事情做,你得勤劳!
    需要注意,由于我们前面为了让后128个线程除了加载之外还有活干,所以一开始blockDim.x*2了,那后面grid stride也如此
    grid stride在这里调整为gridDim.x * blocksize * 2
    同时你的grid那个数字,别用(N+BLOCKSIZE-1)/BLOCKSIZE来获取了,直接键入256,512,1024,2048这些
 
    经过我的测试:512(2^18==262144)
                128(65536==2^16)
                512(2^20==1048576)
                这个block取数也跟N的大小有关,目前来看N大一点的话,512会比较好
 */

const int N = 1048576;
const int BLOCKSIZE = 256;
const int GRIDSIZE = 512;
const int nRepeats = 10;
const int MASK = 0xffffffff;
const int WARPNUM = BLOCKSIZE/32;
#define WARPSIZE 32

template<unsigned int blocksize>
__device__ float unroll_last_warp(float sum){
    if(blocksize>=32) sum += __shfl_down_sync(MASK,sum,16,warpSize); 
    if(blocksize>=16) sum += __shfl_down_sync(MASK,sum,8,warpSize);
    if(blocksize>=8) sum+= __shfl_down_sync(MASK,sum,4,warpSize);
    if(blocksize>=4) sum+= __shfl_down_sync(MASK,sum,2,warpSize);
    if(blocksize>=2) sum+= __shfl_down_sync(MASK,sum,1,warpSize);
    return sum;
}

template<unsigned int blocksize>
__global__ void reduceV2(float* src,float* dst,int n){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = tx + bx * blockDim.x*2;
    int gridStride = blocksize * gridDim.x * 2;
    float sum = 0.0f;
    for(;tid<n;tid+=gridStride){
        // smemTmp[tx] = src[tid] + src[tid+blocksize]; // 某些情况会越界,不想用if,通过?:来实现,避免warp divergence
        float tb = tid+blocksize<n?src[tid+blocksize]:0.0f;
        sum += src[tid] + tb;
    }
    __syncthreads();
    // printf("sum:%lf\n",sum);
    extern __shared__ float smemWarp[];

    const int warpId = tx / warpSize;
    // const int laneId = tx & (warpSize-1);    
    const int laneId = tx % warpSize;

    sum = unroll_last_warp<blocksize>(sum);
    if(laneId == 0)
        smemWarp[warpId] = sum;
    __syncthreads();

    // const int warpNum = blocksize / warpSize;
    sum = (tx < (blocksize/WARPSIZE)) ? smemWarp[laneId] : 0.0f;
    if(warpId == 0){
        sum = unroll_last_warp<blocksize/WARPSIZE>(sum);
    }

    if(tx==0)
        dst[bx] = sum;
    // if(tx == 0)  // 用来校验算法是否错误的
    //     atomicAdd(dst,sum);
}

void testPerfomance(float* dSrc){
    dim3 block(BLOCKSIZE);
    // dim3 grid(((N+BLOCKSIZE-1)/BLOCKSIZE)>>1);
    dim3 grid(GRIDSIZE);

    float allTime=.0f,avgTime=.0f;
    
    size_t sizeDst = sizeof(float) * (GRIDSIZE);
    float* dDst,*hDst;

    hDst = (float*)malloc(sizeDst);
    cudaMalloc(&dDst,sizeDst);

    const int NUM_PER_BLOCK = N/GRIDSIZE;
    const int NUM_PER_THREAD = NUM_PER_BLOCK / BLOCKSIZE;

    for(int i=0;i<nRepeats;i++){
        cudaEvent_t start,end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaMemset(dDst,0,sizeDst);
        memset(hDst,0,sizeDst);

        cudaEventRecord(start);
       
        reduceV2<BLOCKSIZE><<<grid,block,sizeof(float)*WARPNUM>>>(dSrc,dDst,N);
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
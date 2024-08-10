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
const int GRIDSIZE = 1024;
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

template<unsigned int NUM_PER_THREAD,unsigned int blocksize>
__global__ void reduceV2(float* src,float* dst,int n){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    // int tid = tx + bx * blocksize *NUM_PER_THREAD;
    int tid = tx + bx * blockDim.x*2;
    extern __shared__ float smemTmp[];
    int gridStride = blocksize * gridDim.x * 2;
    smemTmp[tx] = 0.0f;
    for(;tid<n;tid+=gridStride){
        // smemTmp[tx] = src[tid] + src[tid+blocksize]; // 某些情况会越界,不想用if,通过?:来实现,避免warp divergence
        float tb = tid+blocksize<n?src[tid+blocksize]:0.0f;
        smemTmp[tx] += src[tid] + tb;
    }

    // for(int iter=0;iter<NUM_PER_THREAD;iter++){
    //     float tb = (tid + iter*blocksize) < n ? src[tid+iter*blocksize] :0.0f;
    //     smemTmp[tx] += tb;
    // }

    __syncthreads();

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

    if(tx==0)
        dst[bx] = smemTmp[0];
    // if(tx == 0)  // 用来校验算法是否错误的
    //     atomicAdd(dst,smemTmp[0]);
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
       
        reduceV2<NUM_PER_THREAD,BLOCKSIZE><<<grid,block,sizeof(float)*BLOCKSIZE>>>(dSrc,dDst,N);
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
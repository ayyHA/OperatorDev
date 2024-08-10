#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARPSIZE 32

#define checkCudaError(func) {              \
    cudaError_t e = (func);                 \
    if(e != cudaSuccess){         \
        printf("CUDA ERROR %s %d : %s\n",__FILE__,__LINE__,cudaGetErrorString(e));  \
        exit(-1);                           \
    }                                       \
}

#define ceilOperation(a,b) ((a+b-1) / b)


/*
    1. 采用wmma实现朴素的sgemm，跟朴素的sgemm实现思路差不多
    不过先前的方法是一个thread处理C中的一个元素，现在是一个warp处理C中的一个tile(16x16的tile,其中A,B的tile的k也是16)
    因为wmma是warp level的mma
*/
__global__ void sgemm_wmma_naive(const half* __restrict__ A, const half* __restrict__ B,
                                 half* __restrict__ C,int M,int N,int K){
    // 沿着K方向上的A,B的tile需要遍历的次数
    const int kCnt = ceilOperation(K,WMMA_K);
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    // 这里是1个block只有1个warp
    const int warpRow = by * WMMA_M;
    const int warpCol = bx * WMMA_N;
    if(warpRow > M || warpCol > N)
        return;
    // 用fragment模板类给数据做包装,C,D阵用accumulator,A,B阵有对应的matrix_a和matrix_b
    wmma::fragment<wmma::accumulator,WMMA_M,WMMA_N,WMMA_K,half> cFrag;

    wmma::fill_fragment(cFrag,0.0);
    for(int k=0;k<kCnt;k++){
        wmma::fragment<wmma::matrix_a,WMMA_M,WMMA_N,WMMA_K,half,wmma::row_major> aFrag;
        wmma::fragment<wmma::matrix_b,WMMA_M,WMMA_N,WMMA_K,half,wmma::col_major> bFrag;

        wmma::load_matrix_sync(aFrag,A+warpRow*K+k*WMMA_K,K);
        wmma::load_matrix_sync(bFrag,B+warpCol*K+k*WMMA_K,K);
    
        wmma::mma_sync(cFrag,aFrag,bFrag,cFrag);
    }
    wmma::store_matrix_sync(C+warpRow*N+warpCol,cFrag,N,wmma::mem_row_major);
    // printf("%f\n",__half2float(C[2]));
    // printf("%f,%f\n",__half2float(A[2]),__half2float(B[2]));
}

int main(int argc,char** argv){ 
    if(argc<4){
        fprintf(stderr,"Please input 4 elements,like: ./xx M N K\n");
        exit(-1);
    }
    int gpu_id = 2;
    cudaDeviceProp prop;

    if(cudaGetDeviceProperties(&prop,gpu_id) == cudaSuccess){
        printf("Using GPU:%d : %s\n",gpu_id,prop.name);
    }

    cudaSetDevice(gpu_id);


    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);


    size_t sizeA = sizeof(half)*M*K;
    size_t sizeB = sizeof(half)*K*N;
    size_t sizeC = sizeof(half)*M*N;

    half* hA = (half*)malloc(sizeA);
    half* hB = (half*)malloc(sizeB);    
    half* hC = (half*)malloc(sizeC);

    for(int i=0;i<M*K;i++){
        hA[i] = i/256.0f;  //i/6;
    }

    for(int i=0;i<K*N;i++){
        hB[i] = (i+1.25f)/128.0f;  //(i+1.25f)/2;
    }
    for(int i=0;i<256;i++)
        printf("hA[%d]: %f; hB[%d]: %f\n",i,__half2float(hA[i]),i,__half2float(hB[i]));

    half* dA,*dB,*dC;
    // float* dC;
    checkCudaError(cudaMalloc(&dA,sizeA));
    checkCudaError(cudaMalloc(&dB,sizeB));
    checkCudaError(cudaMalloc(&dC,sizeC));

    checkCudaError(cudaMemcpy(dA,hA,sizeA,cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dB,hB,sizeB,cudaMemcpyHostToDevice));

    dim3 block(WARPSIZE);
    dim3 grid(ceilOperation(N,WMMA_N),ceilOperation(M,WMMA_M));

    sgemm_wmma_naive<<<grid,block>>>(dA,dB,dC,M,N,K);
    cudaDeviceSynchronize();
    checkCudaError(cudaMemcpy(hC,dC,sizeC,cudaMemcpyDeviceToHost));

    for(int i=0;i<10;i++)
        printf("hC[%d]:[%.6f]\n",i,__half2float(hC[i]));
    half a = 65519;// 明明是65504,为啥到65520才INF,65519都会弄成65504
    printf("test: %f\n",__half2float(a));

    checkCudaError(cudaFree(dA));
    checkCudaError(cudaFree(dB));
    checkCudaError(cudaFree(dC));
    free(hA);
    free(hB);
    free(hC);
    cudaDeviceReset();
    return 0;   
}
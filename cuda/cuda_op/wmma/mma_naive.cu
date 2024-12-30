#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>

// M16N8K16 采用SM86架构进行计算
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define WARPSIZE 32
#define checkCudaError(func) {                                                          \
    cudaError_t e = (func);                                                             \
    if(e != cudaSuccess){                                                               \
        printf("CUDA ERROR %s %d : %s\n",__FILE__,__LINE__,cudaGetErrorString(e));      \
    }                                                                                   \
}

#define ceilOperation(a,b) (((a)+(b)-1)/(b))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define INT4(pointer) (reinterpret_cast<int4*>(&(pointer))[0])

#define LDMATRIX_X1(R0,ADDR)                                                            \
    __asm__ volatile(                                                                   \
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0},[%1]; \n"                               \
        : "=r"(R0)                                                                      \
        : "r"(ADDR))                                                                                   

#define LDMATRIX_X2(R0,R1,ADDR)                                                         \
    __asm__ volatile(                                                                   \
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2]; \n"                            \
        : "=r"(R0),"=r"(R1)                                                             \
        : "r"(ADDR))                                                                                   


#define LDMATRIX_X4(R0,R1,R2,R3,ADDR)                                                   \
    __asm__ volatile(                                                                   \
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4]; \n"                      \
        : "=r"(R0),"=r"(R1),"=r"(R2),"=r"(R3)                                          \
        : "r"(ADDR))                                                                                   

#define HMMA16816(RD0,RD1,RA0,RA1,RA2,RA3,RB0,RB1,RC0,RC1)                                              \
    __asm__ volatile(                                                                                   \
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};  \n"  \
        : "=r"(RD0),"=r"(RD1)                                                                           \
        : "r"(RA0),"r"(RA1),"r"(RA2),"r"(RA3),                                                          \
          "r"(RB0),"r"(RB1),"r"(RC0),"r"(RC1))                                                                                                   

__global__ void mma_naive(half* A, half* B,half* C,const int M,const int N,const int K){
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    // 1个block 1个warp
    const int wcol = bx * MMA_N;
    const int wrow = by * MMA_M;
    const int laneId = threadIdx.x % WARPSIZE;

    if(wrow < M && wcol < N){
        __shared__ half sAtile[MMA_M][MMA_K];
        __shared__ half sBtile[MMA_N][MMA_K];
        __shared__ half sCtile[MMA_M][MMA_N];

        uint32_t aReg[4];
        uint32_t bReg[2];
        uint32_t cReg[2] = {0,0};

        int kTimes = ceilOperation(K,MMA_K);
        
        for(int i=0;i<kTimes;i++){
            // A: gmem->smem 需要注意，赋值时已按照行主序赋值
            INT4(sAtile[laneId/2][(laneId%2)*8]) = INT4(A[(wrow + laneId/2)*K+i*MMA_K +(laneId%2)*8]);
            // B: gmem->smem 需要注意，赋值时已按照列主序赋值
            if(laneId < 16){
            //    INT4(sBtile[laneId/2][(laneId%2)*8]) = INT4(B[(wcol+laneId/2)*K + i*MMA_K + (laneId%2)*8]);
                INT4(sBtile[laneId/2][(laneId%2)*8]) = INT4(B[wcol+(i*MMA_K+laneId)*N]);
            }
            __syncthreads();

            // sAtile取地址     // WHY GET ADDR LIKE THIS?
            uint32_t sAaddr = __cvta_generic_to_shared(&sAtile[laneId%16][(laneId/16)*8]);
            // smem->reg
            LDMATRIX_X4(aReg[0],aReg[1],aReg[2],aReg[3],sAaddr);
            // sBtile取地址
            uint32_t sBaddr = __cvta_generic_to_shared(&sBtile[laneId%8][(laneId/8)*8]);
            // smem->reg        // T16-T31?
            LDMATRIX_X2(bReg[0],bReg[1],sBaddr);

            // hmma
            HMMA16816(cReg[0],cReg[1],aReg[0],aReg[1],aReg[2],aReg[3],bReg[0],bReg[1],cReg[0],cReg[1]);
            __syncthreads();    // WHY SYNC HERE?
        }
        *(reinterpret_cast<uint32_t*>(&sCtile[laneId/4][(laneId%4)*2])) = cReg[0];
        *(reinterpret_cast<uint32_t*>(&sCtile[(laneId/4) + 8][(laneId%4)*2])) = cReg[1];

        __syncthreads();
        
        if(laneId<16){
            INT4(C[(wrow+laneId)*N+wcol]) = INT4(sCtile[laneId][0]);
        }

    }
}

int main(int argc,char** argv){
    // ./mma_naive 256 256 256
    if(argc < 4){
        fprintf(stderr,"Please input 4 elements,like: ./mma_naive M N K\n");
        exit(-1);
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    size_t sizeA = sizeof(half)*M*K;
    size_t sizeB = sizeof(half)*K*N;
    size_t sizeC = sizeof(half)*M*N;

    half* hA =(half*)malloc(sizeA);
    half* hB =(half*)malloc(sizeB);
    half* hC =(half*)malloc(sizeC);

    // 行主序，一行元素是[0,K)这么个大小的元素排列
    for(int i=0;i<M*K;i++){
        // hA[i] = (i%K)/32.f;
        hA[i] = i%16;
    }
    // 列主序，一列元素是[0,K)这么个大小的元素排列
    int cnt = 15;
    for(int i=0;i<K*N;i++){
        // hB[(i%K)*K+i/K] = (K-(i%K))/32.f;
        hB[i] = cnt;
        if(i%N==0){
            cnt--;
            if(cnt==-1)
                cnt = 15;
        }
    }

    half* dA,*dB,*dC;
    checkCudaError(cudaMalloc(&dA,sizeA));
    checkCudaError(cudaMalloc(&dB,sizeB));
    checkCudaError(cudaMalloc(&dC,sizeC));

    checkCudaError(cudaMemcpy(dA,hA,sizeA,cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dB,hB,sizeB,cudaMemcpyHostToDevice));

    dim3 blockDim3(WARPSIZE);
    dim3 gridDim3(ceilOperation(N,MMA_N),ceilOperation(M,MMA_M));

    mma_naive<<<gridDim3,blockDim3>>>(dA,dB,dC,M,N,K);
    checkCudaError(cudaMemcpy(hC,dC,sizeC,cudaMemcpyDeviceToHost));

    // 校验正确性
    // for(int i=0;i<M*N;i++){
    //     printf("C[%03d][%03d]:%f\n",i/N,i%N,__half2float(hC[i]));
    // }

    // 释放内存
    checkCudaError(cudaFree(dA));
    checkCudaError(cudaFree(dB));
    checkCudaError(cudaFree(dC));
    free(hA);
    free(hB);
    free(hC);
    return 0;
}
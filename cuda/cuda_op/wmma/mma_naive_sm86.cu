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
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.b16 {%0},[%1]; \n" : "=r"(R0) : "r"(ADDR))

#define LDMATRIX_X2(R0,R1,ADDR)                                                         \
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1},[%2]; \n" : "=r"(R0),"=r"(R1) : "r"(ADDR))


#define LDMATRIX_X4(R0,R1,R2,R3,ADDR)                                                   \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4]; \n" : "=r"(R0),"=r"(R1),"=r"(R2),"=r"(R3) : "r"(ADDR))

#define HMMA16816(RD0,RD1,RA0,RA1,RA2,RA3,RB0,RB1,RC0,RC1)                                                          \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};  \n"     \ 
        : "=r"(RD0),"=r"(RD1)                                                                                       \
        : "r"(RA0),"r"(RA1),"r"(RA2),"r"(RA3),"r"(RB0),"r"(RB1),"r"(RC0),"r"(RC1))


__global__ void mma_naive(half* A, half* B,half* C,const int M,const int N,const int K){
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
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
               // INT4(sBtile[laneId/2][(laneId%2)*8]) = INT4(B[(wcol+laneId/2)*K + i*MMA_K + (laneId%2)*8]);
                INT4(sBtile[laneId/2][(laneId%2)*8]) = INT4(B[wcol+(i*MMA_K+laneId)*N]);
            }
            __syncthreads();

            // sAtile取地址     // WHY GET ADDR LIKE THIS?
            uint32_t sAaddr = __cvta_generic_to_shared(&sAtile[laneId%16][(laneId/16)*8]);
            // smem->reg
            LDMATRIX_X4(aReg[0],aReg[1],aReg[2],aReg[3],sAaddr);
            // sBtile取地址
            uint32_t sBaddr = __cvta_generic_to_shared(&sBtile[laneId%8][((laneId/8)%2)*8]);
            // smem->reg        // T16-T31?
            LDMATRIX_X2(bReg[0],bReg[1],sBaddr);

            // HMMA
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

/*
__global__ void mma_naive(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                               size_t N, size_t K) {
    const size_t K_tiles = ceilOperation(K, MMA_K);

    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;

    if (warp_row >= M || warp_col >= N) {
        return;
    }

    __shared__ half A_smem[MMA_M][MMA_K];
    __shared__ half B_smem[MMA_N][MMA_K];
    __shared__ half C_smem[MMA_M][MMA_N];

    const size_t lane_id = threadIdx.x % WARPSIZE;

    uint32_t RC[2] = {0, 0};

    #pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        *((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) =
            *((int4 *)(&A[(warp_row + lane_id / 2) * K + i * MMA_K]) + lane_id % 2);

        if (lane_id < MMA_N * 2) {
           // *((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2) =
           //     *((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2);
                *(int4*)(&B_smem[lane_id/2][(lane_id%2)*8]) = *(int4*)(&B[warp_col+(i*MMA_K+lane_id)*N]);
        }

        __syncthreads();

        uint32_t RA[4];
        uint32_t RB[2];

        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
        LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);

        __syncthreads();
    }

    *((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0];
    *((uint32_t *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];

    __syncthreads();

    if (lane_id < MMA_M) {
        *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) = *((int4 *)(&C_smem[lane_id][0]));
    }
}
*/


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
        hA[i] = 1;
    }
    // 列主序，一列元素是[0,K)这么个大小的元素排列
    for(int i=0;i<K*N;i++){
        // hB[(i%K)*K+i/K] = (K-(i%K))/32.f;
        hB[i] = 1;
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

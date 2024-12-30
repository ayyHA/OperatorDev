#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cute/tensor.hpp>
#include <cstdlib>
using namespace cute;

template<typename T,int kTileM,int kTileN,int kTileK,typename TiledMMA>
__global__ void sgemm_v1(T* Cptr,T* Aptr,T* Bptr,int m,int n,int k){
    // 构建Tensor,包装指针
    Tensor gA = make_tensor(make_gmem_ptr(Aptr),make_shape(m,k),make_stride(k,Int<1>{}));
    Tensor gB = make_tensor(make_gmem_ptr(Bptr),make_shape(n,k),make_stride(k,Int<1>{}));
    Tensor gC = make_tensor(make_gmem_ptr(Cptr),make_shape(m,n),make_stride(n,Int<1>{}));
    // local_tile分到自己block的块
    int bx = blockIdx.x;
    int by = blockIdx.y;
    Tensor bA = local_tile(gA,make_shape(Int<kTileM>{},Int<kTileK>{}),make_coord(by,_));    // (kTileM,kTileK,num_ktile)
    Tensor bB = local_tile(gB,make_shape(Int<kTileN>{},Int<kTileK>{}),make_coord(bx,_));    // (kTileN,kTileK,num_ktile)
    Tensor bC = local_tile(gC,make_shape(Int<kTileM>{},Int<kTileN>{}),make_coord(by,bx));   // (kTileM,kTileN)
    // 用TiledMMA,获得每个线程要负责的任务,通过get_slice()和partition_A()和partition_fragment_A()一套组合拳
    TiledMMA tiledMMA;
    auto thr_mma = tiledMMA.get_slice(threadIdx.x);
    auto tAg = thr_mma.partition_A(bA);     // (MMA,MMA_M,MMA_K,num_ktile)
    auto tBg = thr_mma.partition_B(bB);     // (MMA,MMA_N,MMA_K,num_ktile)
    auto tCg = thr_mma.partition_C(bC);     // (MMA,MMA_M,MMA_N)
    // 开寄存器空间
    auto tAr = thr_mma.partition_fragment_A(bA(_,_,0)); // (MMA,MMA_M,MMA_K)
    auto tBr = thr_mma.partition_fragment_B(bB(_,_,0)); // (MMA,MMA_N,MMA_K)
    auto tCr = thr_mma.partition_fragment_C(bC(_,_));   // (MMA,MMA_M,MMA_N)
    // C寄存器清零
    clear(tCr);

    int num_ktile = size<2>(bA);
    #pragma unroll
    for(int i=0;i<num_ktile;i++){
        copy(tAg(_,_,_,i),tAr);
        copy(tBg(_,_,_,i),tBr);
        cute::gemm(tiledMMA,tCr,tAr,tBr,tCr);
    }
    copy(tCr,tCg);
}

int main(){
    const int M = 81920, N = 256, K = 256;
    const int kTileM = 128, kTileN = 128, kTileK = 32;
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    
    using tiledMMA = decltype(make_tiled_mma(
        mma_atom{},
        make_layout(Shape<_2,_2,_1>{}), // AtomMNKLayout
        make_layout(Shape<_1,_2,_1>{})  // ValueMNKLayout
    ));

    dim3 block(size(tiledMMA{}));
    dim3 grid(N/kTileN,M/kTileM);

    half* hAptr,*hBptr,*hCptr;
    size_t sizeA = M*K*sizeof(half);
    size_t sizeB = N*K*sizeof(half);
    size_t sizeC = M*N*sizeof(half);
    hAptr = (half*)malloc(sizeA);
    hBptr = (half*)malloc(sizeB);
    hCptr = (half*)malloc(sizeC);
    for(int i=0;i<M*K;i++){
        hAptr[i] = drand48();
    }
    for(int i=0;i<M*K;i++){
        hBptr[i] = drand48();
    }
    half* dAptr,*dBptr,*dCptr;
    cudaMalloc(&dAptr,sizeA);
    cudaMalloc(&dBptr,sizeB);
    cudaMalloc(&dCptr,sizeC);
    cudaMemcpy(dAptr,hAptr,sizeA,cudaMemcpyHostToDevice);
    cudaMemcpy(dBptr,hBptr,sizeB,cudaMemcpyHostToDevice);
    cudaMemset(dCptr,0,sizeC);
    
    sgemm_v1<half,kTileM,kTileN,kTileK,tiledMMA><<<grid,block>>>(dCptr,dAptr,dBptr,M,N,K);
}
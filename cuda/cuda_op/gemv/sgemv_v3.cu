#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/*
    n == 16
    
    3. sgemv: y = Ax;
    A matrix m x n
    x vector n x 1 
    y vector m x 1

    注意这里的情况跟前面有所不同，现在n越来越小，我们不需要一整个warp的线程就能处理完一行元素，
    比如这里1个warp可以处理两行(即n==16)，因此假定我们的block依旧是(32,4)的情况，那所需的block数目是与1个block能处理的行数挂钩的，所以grid的参数设置需要改变;
    同时内部计算行数的方式也需要改变: bx * blockDim.y * 2(每个warp处理的行数),定位到具体的行(这里指的是warp处理的2行)需要ty*2 warpRow
    然后计算出当前多少个线程1行：nowWarp = warpSize/2 (1个warp,2行) ,由此以tx/nowWarp来计算出nowWarpRow,同时tx%nowWarp可以计算出nowWarpCol，便可以计算了~
    需要注意：这时候到warpSum里去求和，这个WarpSize就是nowWarp了
*/

#define checkCudaError(func) {          \
    cudaError_t e = (func);             \
    if(e != cudaSuccess){               \
        printf("CUDA ERROR %s %d : %s",__FILE__,__LINE__,cudaGetErrorString(e));    \
    }                                   \
}
#define THREAD_PER_BLOCK 128
#define WARPSIZE 32
#define MASK 0xffffffff

template<unsigned int WarpSize>
__device__ __forceinline__ float warpSum(float sum){
    if(WarpSize>=32) sum += __shfl_down_sync(MASK,sum,16);
    if(WarpSize>=16) sum += __shfl_down_sync(MASK,sum,8);
    if(WarpSize>=8) sum += __shfl_down_sync(MASK,sum,4);
    if(WarpSize>=4) sum += __shfl_down_sync(MASK,sum,2);
    if(WarpSize>=2) sum += __shfl_down_sync(MASK,sum,1);
    return sum;
}

template<unsigned int ROW_PER_WARP>
__global__ void sgemv_v3(float* __restrict__ A,float* __restrict__ x, float* __restrict__ y,const int m,const int n){
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    int warpRow = bx*blockDim.y*ROW_PER_WARP + ty*ROW_PER_WARP;
    const int tmpWarpSize = WARPSIZE / ROW_PER_WARP;  // 目的1行1个warp，当n太小，所以调整warpSize为tmpWarpSize
    int tmpWarpRow = warpRow + tx/tmpWarpSize;
    int tmpCol = tx %tmpWarpSize;

    if(tmpWarpRow<m){
        float sum = .0f;
        sum += A[tmpWarpRow*n + tmpCol] * x[tmpCol];
        sum = warpSum<tmpWarpSize>(sum);
        if(tmpCol == 0)
            y[tmpWarpRow] = sum;
    }
}


int main(int argc,char** argv){
    if(argc < 2){
        // ./sgemv_v3 16384
        printf("Please input 2 argument, like: ./file M \n");
        exit(-1);
    }

    int m = atoi(argv[1]);
    const int n = 16;

    int gpu_id = 0;
    cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop,gpu_id) == cudaSuccess){
        printf("Use GPU:%d,info:%s\n",gpu_id,prop.name);
    }

    cudaSetDevice(gpu_id);

    // 开空间给A,x,y
    size_t sizeA = sizeof(float)*m*n;
    size_t sizeX = sizeof(float)*n;
    size_t sizeY = sizeof(float)*m;

    float* hA = (float*)malloc(sizeA);
    float* hX = (float*)malloc(sizeX);
    float* hY = (float*)malloc(sizeY);
    float* hYblas = (float*)malloc(sizeY);
    // 随机初始化A,x
    for(int i=0;i<m*n;i++){
        hA[i] = (i+1)/3.0f;
    }
    for(int i=0;i<n;i++){
        hX[i] = i+8.2f;
    }

    // 开空间给device
    float* dA,*dX,*dY;
    checkCudaError(cudaMalloc(&dA,sizeA));
    checkCudaError(cudaMalloc(&dX,sizeX));
    checkCudaError(cudaMalloc(&dY,sizeY));

    checkCudaError(cudaMemcpy(dA,hA,sizeA,cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dX,hX,sizeX,cudaMemcpyHostToDevice));

    checkCudaError(cudaMemset(dY,0,sizeY));
    int nRepeats = 1000;
    
    cudaEvent_t start,stop;
    float elapsed_time;
    checkCudaError(cudaEventCreate(&start));    
    checkCudaError(cudaEventCreate(&stop));    

    const int ROW_PER_WARP = WARPSIZE/n;
    const int WARP_PER_BLOCK = THREAD_PER_BLOCK/WARPSIZE;
    const int ROW_PER_BLOCK = ROW_PER_WARP * WARP_PER_BLOCK;
    const int NUM_BLOCK = m/ROW_PER_BLOCK;


    dim3 grid(NUM_BLOCK);
    // dim3 block(4,32); // 这个对着sgemv_v1,思考有点倒过来了
    dim3 block(WARPSIZE,WARP_PER_BLOCK);

    // kernel start
    checkCudaError(cudaEventRecord(start));
    for(int i=0;i<nRepeats;i++){
        sgemv_v3<ROW_PER_WARP><<<grid,block>>>(dA,dX,dY,m,n);
    }
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&elapsed_time,start,stop));

    checkCudaError(cudaMemcpy(hY,dY,sizeY,cudaMemcpyDeviceToHost));

    elapsed_time = elapsed_time / 1000; // ms -> s
    elapsed_time = elapsed_time / nRepeats; // avg time

    printf("SGEMV_V3: [%.9f]s\n",elapsed_time);
    // printf("SGEMV_V1: [%.9f]s\n",elapsed_time);
    
    // cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f,beta = 0.0f;
    float elapsed_time_blas;

    checkCudaError(cudaMemset(dY,0,sizeY));
    checkCudaError(cudaEventRecord(start));
    for(int i=0;i<nRepeats;i++){
        cublasSgemv(
            handle,CUBLAS_OP_T,
            n,m,&alpha,
            dA,n,
            dX,1,&beta,
            dY,1
        );
    }
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventElapsedTime(&elapsed_time_blas,start,stop));
    
    checkCudaError(cudaMemcpy(hYblas,dY,sizeY,cudaMemcpyDeviceToHost));

    elapsed_time_blas = elapsed_time_blas / 1000;
    elapsed_time_blas = elapsed_time_blas / nRepeats;
    printf("CUSGEMV: [%.9f]s\n",elapsed_time_blas);

    // isTrue
    // 我的这个有错
    // float maxError = 0.0f;
    // for(int i=0;i<m;i++){
    //     maxError = fmax(maxError,fabs(hY[i] - hYblas[i]));
    // }
    // printf("MaxError: %.6f\n",maxError);
    // printf("Result: %s\n",maxError<1e-6? "PASS" : "FAILED");
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < m; i++) {
        double abs_err = fabs(hY[i] - hYblas[i]);
        double dot_length = m;
        double abs_val = fabs(hY[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, hY[i], hYblas[i], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");

    
    for(int i=0;i<10;i++){
        printf("[hY%d]=%.6f , [hYblas%d]=%.6f\n",i,hY[i],i,hYblas[i]);
    }

    // clean unused space
    free(hA);
    free(hX);
    free(hY);
    checkCudaError(cudaFree(dA));
    checkCudaError(cudaFree(dX));
    checkCudaError(cudaFree(dY));
    checkCudaError(cudaEventDestroy(start));
    checkCudaError(cudaEventDestroy(stop));
    cublasDestroy(handle);
    cudaDeviceReset();
    return 0;
}
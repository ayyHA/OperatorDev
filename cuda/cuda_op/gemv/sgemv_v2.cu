#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

//  n == 128
/**
    2. sgemv: y = Ax;
    A matrix m x n
    x vector n x 1 
    y vector m x 1

    跟之前一样，1个warp负责1行的计算
    gemv的优化与shape有关，这里考虑的是n==128的情况
    可以采用向量化访存，float4嘛
 */

#define checkCudaError(func) {          \
    cudaError_t e = (func);             \
    if(e != cudaSuccess){               \
        printf("CUDA ERROR %s %d : %s",__FILE__,__LINE__,cudaGetErrorString(e));    \
    }                                   \
}

#define MASK 0xffffffff
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


template<unsigned int WarpSize>
__device__ __forceinline__ float warpSum(float sum){
    if(WarpSize>=32) sum += __shfl_down_sync(MASK,sum,16);
    if(WarpSize>=16) sum += __shfl_down_sync(MASK,sum,8);
    if(WarpSize>=8) sum += __shfl_down_sync(MASK,sum,4);
    if(WarpSize>=4) sum += __shfl_down_sync(MASK,sum,2);
    if(WarpSize>=2) sum += __shfl_down_sync(MASK,sum,1);
    return sum;
}

__global__ void sgemv_v2(float* __restrict__ A,float* __restrict__ x,float* __restrict__ y,int m,int n){
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int WarpSize =32;
    int row = bx*blockDim.y + ty;
    int col = tx;
    
    A = &A[row*n];
    
    float sum = 0.0f;
    if(row<m){
        int nIterations = ((n+WarpSize-1)/WarpSize)/4; 
        int stride = WarpSize;
        for(int i=0;i<nIterations;i++){
            float4 a4 = reinterpret_cast<float4*>(A)[col];//FLOAT4(A[col]);这种写法是错的，会misaligned，具体原因：
            /*
                [0,1,2,3,4,5,6,7,8,...]  每个元素4B
                假设FLOAT4(A[0])则是[0,1,2,3],FLOAT4(A[1])则是[1,2,3,4]
                不是我们期望的：[4,5,6,7],感觉就是，没有16B对齐，这只挪动了4B的位置
            */
            float4 x4 = reinterpret_cast<float4*>(x)[col];//FLOAT4(x[col]);
            sum += a4.x * x4.x;
            sum += a4.y * x4.y;
            sum += a4.z * x4.z;
            sum += a4.w * x4.w;
            col += stride;
        }
        sum = warpSum<WarpSize>(sum);
        if(tx ==  0){
            y[row] = sum;
        }
    }
}


int main(int argc,char** argv){
    if(argc < 3){
        // ./sgemv_v2 16384 128
        printf("Please input 3 argument, like: ./file M N\n");
        exit(-1);
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);

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

    dim3 grid((m+3)/4);
    // dim3 block(4,32); // 这个对着sgemv_v1,思考有点倒过来了
    dim3 block(32,4);

    // kernel start
    checkCudaError(cudaEventRecord(start));
    for(int i=0;i<nRepeats;i++){
        sgemv_v2<<<grid,block>>>(dA,dX,dY,m,n);
    }
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&elapsed_time,start,stop));

    checkCudaError(cudaMemcpy(hY,dY,sizeY,cudaMemcpyDeviceToHost));

    elapsed_time = elapsed_time / 1000; // ms -> s
    elapsed_time = elapsed_time / nRepeats; // avg time

    printf("SGEMV_V0: [%.9f]s\n",elapsed_time);
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
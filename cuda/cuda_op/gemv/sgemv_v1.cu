#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/*
//  n == 32

    1. sgemv: y = Ax;
    A matrix mxn
    x vector nx1 
    y vector mx1

    gemv的优化与shape有关,首先考虑一个n==32的情况
    根据上面对gemv的描述，可知：这里可以用一个warp负责一行的计算，那也即是说，有多少行就相当于有多少个warp，
    当然我们分配执行参数的时候是根据grid,block来进行分配的，我们也知道block数目多可以多利用sm,warp数目多，可以在一个sm内多切换
    这些可以遮掩访存的开销，并利用GPU的更多资源(sm)

    那么假设定下一个block是4个warp，则block(32,4)
    那可以算出需要的block数目，用以放于grid中grid(((m+3))/4) // m+3向上取整，/4是一个block处理4行，需要的block数目
    然后gemv本质上是：
    A[row*n + col] * x[col]：
        row的变化是根据warpId和block而变化的，我们可以先定位到某一个具体的block(bx*blockDim.y)，计算出对应的warpId(ty)，负责对应的行，比如warp1负责第一行的数与x的点积,warp5负责第5行与x的点积
        col的变化是某一个warp内的线程的landId处理对应的(landId=tx%warpSize)
        同时这个可以扩展到n>32的情况，因为每个线程负责对应的数，步距是warpSize即可，但是扩展到如n==64或是n==128的情况，可以采用向量化访存来进行优化
        计算完后，我们每个线程有个sum得到的是它跟对应数的乘积（如果n>32则是乘积和）,那么要得到y[row]的值，还要把它们加起来，因此可以用束内归约的shuffle来实现
        当计算完毕后，laneId==0的则把它的sum值放到当前y[row]中，由此则完成计算
    
 */

#define checkCudaError(func) {          \
    cudaError_t e = (func);             \
    if(e != cudaSuccess){               \
        printf("CUDA ERROR %s %d : %s",__FILE__,__LINE__,cudaGetErrorString(e));    \
    }                                   \
}

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

__global__ void sgemv_v0(float* __restrict__ A,float* __restrict__ x,float* __restrict__ y,int m,int n){
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int row = bx*blockDim.y + ty;
    int col = tx;
    
    float sum=0.0f;
    if(row<m){
        int nIterations = (n+warpSize-1)/warpSize;
        #pragma unroll
        for(int i=0;i<nIterations;i++){
            sum += A[row*n+col] * x[col];
            col += warpSize;
        }
    }

    int laneId = tx % warpSize;
    const int WarpSize = 32;
    sum = warpSum<WarpSize>(sum);
    if(laneId == 0){
        y[row] = sum;
    }
}

__global__ void sgemv_v1(float* __restrict__ A,float* __restrict__ x,float* __restrict__ y,int m,int n){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    int row = bx*blockDim.x + tx;
    int col = ty;

    if(row<m){
        int nIterations = n/warpSize;
        if(nIterations==0)
            nIterations = 1;
        float sum =0.0f;
        #pragma unroll
        for(int i=0;i<nIterations;i++){
            // if(col<n){
                sum += A[row*n + col] * x[col];
                col += warpSize;
            // }
        }
        const int WarpSize = 32;
        sum = warpSum<WarpSize>(sum);
        int laneId = tx%32;
        if(laneId==0){
            y[row] = sum;
        } 
    }
}

int main(int argc,char** argv){
    if(argc < 3){
        // ./sgemv_v1 16384 32
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
        // sgemv_v1<<<grid,block>>>(dA,dX,dY,m,n);
        sgemv_v0<<<grid,block>>>(dA,dX,dY,m,n);
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
    // float maxError = 0.0f;
    // for(int i=0;i<m;i++){
    //     maxError = fmax(maxError,fabs(hY[i] - hYblas[i]));
    // }
    // printf("Result: %s\n",maxError<1e-6? "PASS" : "FAILED");
    // for(int i=0;i<10;i++){
    //     printf("[hY%d]=%.6f , [hYblas%d]=%.6f\n",i,hY[i],i,hYblas[i]);
    // }
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
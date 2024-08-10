#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define checkCudaError(func) {  \
    cudaError_t e = (func);     \
    if(e != cudaSuccess){       \
        printf("CUDA ERROR %s %d : %s",__FILE__,__LINE__,cudaGetErrorString(e));    \
    }                           \
}
#define FLOAT2(pointer) (reinterpret_cast<float2*>((&pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

const int n = 32*1024*1024;
const int BLOCKSIZE = 256;

__global__ void elementwiseAddOrigin(float* src1,float* src2,float* dst){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    dst[tid] = src1[tid] + src2[tid];
}

__global__ void elementwiseAddFloat2(float* src1,float* src2,float* dst){
    int tid = (threadIdx.x + blockIdx.x * blockDim.x)*2;
    float2 s1 = FLOAT2(src1[tid]);
    float2 s2 = FLOAT2(src2[tid]);
    float2 dt;
    dt.x = s1.x + s2.x;
    dt.y = s1.y + s2.y;
    FLOAT2(dst[tid]) = dt;
}

__global__ void elementwiseAddFloat4(float* src1,float* src2,float* dst){
    int tid =  (threadIdx.x + blockIdx.x * blockDim.x)*4;
    float4 s1 = FLOAT4(src1[tid]);
    float4 s2 = FLOAT4(src2[tid]);
    float4 dt;
    dt.x = s1.x + s2.x;
    dt.y = s1.y + s2.y;
    dt.z = s1.z + s2.z;
    dt.w = s1.w + s2.w;
    FLOAT4(dst[tid]) = dt;
}

int main(int argc,char** argv){
    int gpu_id = 1;
    cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop,gpu_id)==cudaSuccess){
        printf("Use GPU:%d:%s\n",gpu_id,prop.name);
    }
    cudaSetDevice(gpu_id);

    size_t sizeN = sizeof(float) * n;
    float* hSrc1 = (float*)malloc(sizeN);
    float* hSrc2 = (float*)malloc(sizeN);
    float* hDst = (float*)malloc(sizeN);
    float* hCheck = (float*)malloc(sizeN);
    
    float* dSrc1,*dSrc2,*dDst;
    checkCudaError(cudaMalloc(&dSrc1,sizeN));
    checkCudaError(cudaMalloc(&dSrc2,sizeN));
    checkCudaError(cudaMalloc(&dDst,sizeN));

    for(int i=0;i<n;i++){
        hSrc1[i] = 1.25;
        hSrc2[i] = i;
        hDst[i] = 1.25f+i;
    }

    checkCudaError(cudaMemcpy(dSrc1,hSrc1,sizeN,cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dSrc2,hSrc2,sizeN,cudaMemcpyHostToDevice));

    // dim3 grid(n/BLOCKSIZE);
    // dim3 grid(n/BLOCKSIZE/2);
    dim3 grid(n/BLOCKSIZE/4);
    dim3 block(BLOCKSIZE);

    cudaEvent_t start,stop;
    float elapsed_time;
    int nRepeats = 2000;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));
    checkCudaError(cudaEventRecord(start));
    for(int i=0;i<nRepeats;i++){
        // elementwiseAddOrigin<<<grid,block>>>(dSrc1,dSrc2,dDst);
        // elementwiseAddFloat2<<<grid,block>>>(dSrc1,dSrc2,dDst);
        elementwiseAddFloat4<<<grid,block>>>(dSrc1,dSrc2,dDst);
    }
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&elapsed_time,start,stop));

    elapsed_time = elapsed_time / 1000;      // ms->s
    elapsed_time = elapsed_time / nRepeats;  // avgTime

    checkCudaError(cudaMemcpy(hCheck,dDst,sizeN,cudaMemcpyDeviceToHost));
    float maxError = 0.0f;
    for(int i=0;i<n;i++){
        maxError = max(maxError,fabs(hCheck[i]-hDst[i]));
        if(maxError!=0.0f){
            printf("ERROR: hCheck[%d]:%.2f hDst[%d]:%.2f\n",i,hCheck[i],i,hDst[i]);
        }
    }
    printf("maxError=%.6f\n",maxError);

    double bandWidth = sizeN*3 / elapsed_time;
    bandWidth /= 1e9;
    printf("%.6lf GB/S \n",bandWidth);
    
    free(hSrc1);
    free(hSrc2);
    free(hDst);
    free(hCheck);
    checkCudaError(cudaFree(dSrc1));
    checkCudaError(cudaFree(dSrc2));
    checkCudaError(cudaFree(dDst));
    checkCudaError(cudaEventDestroy(start));
    checkCudaError(cudaEventDestroy(stop));

    cudaDeviceReset();
    return 0;
}
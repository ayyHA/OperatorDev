#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>

float serialReduce(float *src, int n)
{
    float psum = 0.0f;
    for (int i = 0; i < n; i++)
        psum += src[i];
    return psum;
}

/* 1. 朴素的归约,interleaved addressing*/
__global__ void reduceV1(float *src, float *dst, int n)
{
    int const bx = blockIdx.x;
    int const tx = threadIdx.x;
    int const tid = tx + bx * blockDim.x;
    extern __shared__ float smemTmp[];
    smemTmp[tx] = tid < n ? src[tid] : 0.0;
    __syncthreads();
    // 两两相加,步距是2次幂
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (tx % (stride << 1) == 0)
        {
            smemTmp[tx] = smemTmp[tx] + smemTmp[tx + stride];
        }
        __syncthreads();
    }
    if (tx == 0)
        dst[bx] = smemTmp[0];
}

/* 2. 考虑到线程束分化会降低执行效率后的归约 */
__global__ void reduceV2(float *src, float *dst, int n)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int tid = bx * blockDim.x + tx;
    extern __shared__ float smemTmp[];
    smemTmp[tx] = tid < n ? src[tid] : 0.0f;
    __syncthreads(); // 同步一下,以便block内的所有线程看到的共享内存一个样
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int idx = stride * 2 * tx;
        if (idx < blockDim.x)
            smemTmp[idx] = smemTmp[idx] + smemTmp[idx + stride];
        __syncthreads();
    }
    if (tx == 0)
        dst[bx] = smemTmp[0];
}


/* 3. continuous addressing to solve smem bank conflict*/
/* 注意:边界存在[0,...,32],即33个线程,最后一个线程累加的值会被忽略 */
__global__ void reduceV3(float *src, float *dst, int n)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int tid = bx * blockDim.x + tx;
    extern __shared__ float smemTmp[];
    smemTmp[tx] = tid < n ? src[tid] : 0.0;
    __syncthreads();

    for (int stride = (blockDim.x >> 1); stride > 0; stride >>= 1)
    {
        if (tx < stride)
            smemTmp[tx] = smemTmp[tx] + smemTmp[tx + stride];
        __syncthreads();
    }

    if (tx == 0)
        dst[bx] = smemTmp[0];
}

__device__ void warpUnroll(volatile float *src, int tx)
{
    src[tx] += src[tx + 32];
    src[tx] += src[tx + 16];
    src[tx] += src[tx + 8];
    src[tx] += src[tx + 4];
    src[tx] += src[tx + 2];
    src[tx] += src[tx + 1];
}

/* 4. the last warp unroll */
__global__ void reduceV4(float *src, float *dst, int n)
{
    int const bx = blockIdx.x;
    int const tx = threadIdx.x;
    int const tid = bx * blockDim.x + tx;
    extern __shared__ float smemTmp[];
    smemTmp[tx] = tid < n ? src[tid] : .0f;
    __syncthreads();

    for (int stride = (blockDim.x >> 1); stride > 32; stride >>= 1)
    {
        if (tx < stride)
            smemTmp[tx] += smemTmp[tx + stride];
        __syncthreads();
    }

    // 用一个warp来做最后的展开
    if (tx < 32)
        warpUnroll(smemTmp, tx);

    if (tx == 0)
        dst[bx] = smemTmp[0];
}

int main()
{
    int n = 1024;    // 1K
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    float *src, *dst;
    size_t sizeSrc = sizeof(float) * n;
    size_t sizeDst = sizeof(float) * grid.x;
    cudaMallocManaged(&src, sizeSrc);
    cudaMallocManaged(&dst, sizeDst);

    for (int i = 0; i < n; i++)
    {
        src[i] = i;
        printf("[%d]:%f\n", i, src[i]);
    }
    float result = serialReduce(src, n);

    reduceV1<<<grid, block, block.x * sizeof(float)>>>(src, dst, n);
    cudaDeviceSynchronize();
    reduceV1<<<1, grid.x, grid.x * sizeof(float)>>>(dst, dst, grid.x);
    cudaDeviceSynchronize();
    float maxError = .0f;
    maxError = fmax(maxError, fabs(dst[0] - result));
    printf("reduceV1 maxError: %f", maxError);

    // reduceV2<<<grid, block, block.x * sizeof(float)>>>(src, dst, n);
    // cudaDeviceSynchronize();
    // reduceV2<<<1, grid.x, grid.x * sizeof(float)>>>(dst, dst, grid.x);
    // cudaDeviceSynchronize();
    // float maxError = .0f;
    // maxError = fmax(maxError, fabs(dst[0] - result));
    // printf("reduceV2 maxError: %f", maxError);

    // 这个取值方式与V1有了很大的变化,但是针对奇数个线程取值边界没处理好
    // reduceV3<<<grid, block, block.x * sizeof(float)>>>(src, dst, n);
    // cudaDeviceSynchronize();
    // reduceV3<<<1, grid.x, grid.x * sizeof(float)>>>(dst, dst, grid.x);
    // cudaDeviceSynchronize();
    // float maxError = .0f;
    // maxError = fmax(maxError, fabs(dst[0] - result));
    // printf("reduceV3 maxError: %f", maxError);

    // reduceV4<<<grid, block, block.x * sizeof(float)>>>(src, dst, n);
    // cudaDeviceSynchronize();
    // reduceV4<<<1, grid.x, grid.x * sizeof(float)>>>(dst, dst, grid.x);
    // cudaDeviceSynchronize();
    // float maxError = .0f;
    // maxError = fmax(maxError, fabs(dst[0] - result));
    // printf("reduceV4 maxError: %f", maxError);

    cudaFree(src);
    cudaFree(dst);
    return 0;
}
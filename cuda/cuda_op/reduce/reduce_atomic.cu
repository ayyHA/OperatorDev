#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

/**
 1. 这里实际上是reduce_origin和reduce_atomic两种方法
    之所以出现两种方法是因为前者需要调用两次核函数,后者通过原子加来实现累和
    但存在部分问题,如前者的第二次核函数因为算法原因,输入的block的大小一定得是2的次幂,不然必错
    第二个问题是原子加和两次核函数调用都会出现一定的误差,很奇怪哪里来的误差
    第三个问题是串行的居然误差更大   

    误差由来可能是浮点数的表示问题,如1.23用阶码尾数表示,天然存在误差 

    这里的核函数存在的问题:
    以第0个warp为例:
    第1次迭代 (2*s):1个warp中只有2的倍数的线程号可以进入if另外的不进入if;
    第2次迭代 (2*s):1个warp中只有4的倍数的线程号可以进入if另外的不进入if;
    ...
    由此可见,发生了线程束分化的问题
 */
const int N = (65536);
// block内thread为256
const int BLOCKSIZE = 256;

float serialReduce(float *src, int n)
{
    float psum = 0.0f;
    for (int i = 0; i < n; i++)
        psum += src[i];
    return psum;
}

__global__ void reduce_origin(float *src, float *dst, int n)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = tid + bid * blockDim.x;
    extern __shared__ float smemTmp[];
    smemTmp[tid] = (i < n) ? src[i] : 0.0f;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (s << 1) == 0)
        {
            smemTmp[tid] = smemTmp[tid] + smemTmp[tid + s];
        }
        __syncthreads();
    }
    // if (tid == 0)
    //     dst[bid] = smemTmp[0];

    if (tid == 0)
        atomicAdd(dst, smemTmp[0]);
}

int main()
{
    dim3 block(BLOCKSIZE);
    dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE);

    float *dSrc, *dDst;
    size_t srcSize = sizeof(float) * N;
    size_t dstSize = sizeof(float) * grid.x;

    cudaMallocManaged(&dSrc, srcSize);
    cudaMallocManaged(&dDst, dstSize);

    for (int i = 0; i < N; i++)
        dSrc[i] = 8.88f;

    float serialAns;
    serialAns = serialReduce(dSrc, N);

    reduce_origin<<<grid, block, BLOCKSIZE * sizeof(float)>>>(dSrc, dDst, N);
    cudaDeviceSynchronize();
    // for (int i = 0; i < N; i++)
    //     printf("[%d]:%.6f\n", i, dDst[i]);

    // 要扩成(向上取)离grid.x最近的2的次幂,不然会有错误,这里没改好,通过原子加的方式处理
    // if (grid.x % 2 != 0)
    //     grid.x = 2 * grid.x - 2;

    // reduce_origin<<<1, grid.x, grid.x * sizeof(float)>>>(dDst, dDst, grid.x);
    // cudaDeviceSynchronize();
    // for (int i = 0; i < grid.x; i++)
    //     printf("[%d]:%.6f\n", i, dDst[i]);
    printf("serialAns:%.6f\ndDst[0]:%.6f\n", serialAns, dDst[0]);

    float maxError = 0.0f;
    maxError = fmax(maxError, fabs(serialAns - dDst[0]));
    printf("maxError: %.6f\n", maxError);

    cudaFree(dSrc);
    cudaFree(dDst);
    return 0;
}
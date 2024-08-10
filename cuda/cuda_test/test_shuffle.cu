#include <cuda_runtime.h>
#include <stdio.h>

/*
已测:
 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
 4  5  6  7  0  1  2  3 12 13 14 15  8  9 10 11 20 21 22 23 16 17 18 19 28 29 30 31 24 25 26 27
*/
#define ARRAY_SIZE 32
#define TEST_SIZE 4
#define MASK 0xffffffff
#define LANEMASK 0x00000001

__global__ void shuffle(int* src,int* dst){
    int idx = threadIdx.x * TEST_SIZE;
    int arr[TEST_SIZE];
    for(int i=0;i<TEST_SIZE;i++)
        arr[i] = src[idx+i];
    __syncwarp();
    arr[0] = __shfl_xor_sync(MASK,arr[0],LANEMASK,warpSize);
    arr[1] = __shfl_xor_sync(MASK,arr[1],LANEMASK,warpSize);
    arr[2] = __shfl_xor_sync(MASK,arr[2],LANEMASK,warpSize);
    arr[3] = __shfl_xor_sync(MASK,arr[3],LANEMASK,warpSize);
    // printf("arr[]:%d",arr[0]);
    for(int i=0;i<TEST_SIZE;i++)
        dst[idx+i] = arr[i];
}


int main(){
    size_t sizeArray = ARRAY_SIZE * sizeof(int);
    int* hSrc = (int*)malloc(sizeArray);
    int* hDst = (int*)malloc(sizeArray);
    int* dSrc,*dDst;
    cudaMalloc(&dSrc,sizeArray);
    cudaMalloc(&dDst,sizeArray);

    for(int i=0;i<ARRAY_SIZE;i++)
        hSrc[i] = i;

    for(int i=0;i<ARRAY_SIZE;i++)
        printf("%2d ",hSrc[i]);
    printf("\n");

    cudaMemcpy(dSrc,hSrc,sizeArray,cudaMemcpyHostToDevice);

    shuffle<<<1,8>>>(dSrc,dDst);

    cudaMemcpy(hDst,dDst,sizeArray,cudaMemcpyDeviceToHost);

    for(int i=0;i<ARRAY_SIZE;i++)
        printf("%2d ",hDst[i]);
    printf("\n");

    cudaFree(dSrc);
    cudaFree(dDst);
    free(hSrc);
    free(hDst);
    return 0;
}
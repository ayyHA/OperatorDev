#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define SHAPE 8*8*4

#define checkCudaError(func) {                                                      \
    cudaError_t e = (func);                                                         \
    if(e!=cudaSuccess){                                                             \
        printf("CUDA ERROR %s %d : %s\n",__FILE__,__LINE__,cudaGetErrorString(e));  \
    }                                                                               \
}

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void test_ldmatrix(half* dIn,half* dOut){
    __shared__ half smemTmp[SHAPE];
    int idx = threadIdx.x * 8;    
    // 把数据从gmem -> smem,一个线程取一行数据，即128bit共8个fp16,也即是4个fp32,可以用float4一次取完
    // FLOAT4(smemTmp[idx]) = FLOAT4(dIn[idx]);
    *(reinterpret_cast<float4*>(&smemTmp[idx])) = *(reinterpret_cast<float4*>(&dIn[idx]));
    // printf("[%d]:[%f]\n",threadIdx.x,__half2float(smemTmp[idx]));    // smem数据是按要求摆放的 
    
    uint32_t reg[4];
    // 使用ldmatrix, smem -> reg
    __asm__ volatile(
        "ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];    \n"
        : "=r"(reg[0]),
          "=r"(reg[1]),
          "=r"(reg[2]),
          "=r"(reg[3])
        : "l"(&smemTmp[idx])
    );

    // 把每个线程寄存器得到的东西给它移动到我们的dOut中 reg -> gmem
    *(reinterpret_cast<float*>(&dOut[idx]))    = *(reinterpret_cast<float*>(&reg[0]));
    *(reinterpret_cast<float*>(&dOut[idx+2]))  = *(reinterpret_cast<float*>(&reg[1]));
    *(reinterpret_cast<float*>(&dOut[idx+4]))  = *(reinterpret_cast<float*>(&reg[2]));
    *(reinterpret_cast<float*>(&dOut[idx+6]))  = *(reinterpret_cast<float*>(&reg[3]));
}


int main(){
    // 数组是 fp16,4个8*8的阵,用以查看ldmatrix中.num为.x4的数据排布
    half hLoad[SHAPE];
    half hStore[SHAPE];
    // 初始化data,为[0,255]的整型
    for(int i=0;i<SHAPE;i++){
        hLoad[i]=i;
    }

    size_t sizeData = sizeof(half) * (SHAPE);
    half* dLoad,*dStore;

    checkCudaError(cudaMalloc(&dLoad,sizeData));
    checkCudaError(cudaMalloc(&dStore,sizeData));

    checkCudaError(cudaMemcpy(dLoad,hLoad,sizeData,cudaMemcpyHostToDevice));
    test_ldmatrix<<<1,32>>>(dLoad,dStore);
    checkCudaError(cudaMemcpy(hStore,dStore,sizeData,cudaMemcpyDeviceToHost));

    for(int i=0;i<32;i++){
        printf("T%02d ",i);
        for(int j=0;j<8;j+=2){
            printf("r%d %12f %12f ",j/2,__half2float(hStore[i*8+j]),__half2float(hStore[i*8+j+1]));
        }
        printf("\n");
    }    

    checkCudaError(cudaFree(dLoad));
    checkCudaError(cudaFree(dStore));
    return 0;
}
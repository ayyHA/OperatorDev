#include <cuda_runtime.h>

__global__ void func(int c,int* a){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] *= c;
}

int main(){
    return 0;
}
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cstdlib>
#include <iostream>

/* z = a*x + b*y + c */
template<int kNumPerThread=8>
__global__ void axpby(half* z,const int n,const half* x,const half* y,const half a,const half b,const half c){
    using namespace cute;
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid>=n/kNumPerThread)
        return;
    // 封装裸指针成Tensor
    Tensor tz = make_tensor(make_gmem_ptr(z),make_shape(n));
    Tensor tx = make_tensor(make_gmem_ptr(x),make_shape(n));
    Tensor ty = make_tensor(make_gmem_ptr(y),make_shape(n));
    // 切块取对应块,此时应该还是堆上内存
    Tensor tz_tile = local_tile(tz,make_shape(Int<kNumPerThread>{}),make_coord(tid));
    Tensor tx_tile = local_tile(tx,make_shape(Int<kNumPerThread>{}),make_coord(tid));
    Tensor ty_tile = local_tile(ty,make_shape(Int<kNumPerThread>{}),make_coord(tid));
    // 开辟栈上内存(寄存器)
    Tensor tz_tile_reg = make_tensor_like(tz_tile);
    Tensor tx_tile_reg = make_tensor_like(tx_tile);
    Tensor ty_tile_reg = make_tensor_like(ty_tile);
    // LDG.128
    copy(tx_tile,tx_tile_reg);
    copy(ty_tile,ty_tile_reg);
    // 以用上FMA
    half2 aa = {a,a};
    half2 bb = {b,b};
    half2 cc = {c,c};
    // 变换tensor的数据类型
    auto tz_tile_reg_2 = recast<half2>(tz_tile_reg);
    auto tx_tile_reg_2 = recast<half2>(tx_tile_reg);
    auto ty_tile_reg_2 = recast<half2>(ty_tile_reg);

    for(int i=0;i<size(tz_tile_reg_2);i++){
        tz_tile_reg_2(i) = aa*tx_tile_reg_2(i) + (bb*ty_tile_reg_2(i) + cc);
    }

    auto tz_tile_regx = recast<half>(tz_tile_reg_2);
    copy(tz_tile_regx,tz_tile);
}

int main(){
    const int n = 256;
    half* hx,*hy,*hz,*hzCheck;
    size_t sizex = sizeof(half) * n;
    hx = (half*)malloc(sizex);
    hy = (half*)malloc(sizex);
    hz = (half*)malloc(sizex);
    // hzCheck = (half*)malloc(sizex);

    half* dx,*dy,*dz;
    cudaMalloc(&dx,sizex);
    cudaMalloc(&dy,sizex);
    cudaMalloc(&dz,sizex);

    for(int i=0;i<n;i++){
        hx[i] = i/2.0;
        hy[i] = i/2.0;
    }
    half a=1.0,b=1.0,c=2.0;

    // for(int i=0;i<n;i++){
    //     hzCheck[i] = a*hx[i] + b*hy[i] + c;
    // }

    cudaMemcpy(dx,hx,sizex,cudaMemcpyHostToDevice);
    cudaMemcpy(dy,hy,sizex,cudaMemcpyHostToDevice);
    cudaMemset(dz,0,sizex);

    dim3 block(32);
    dim3 grid(1);
    axpby<<<grid,block>>>(dz,n,dx,dy,a,b,c);
    cudaMemcpy(hz,dz,sizex,cudaMemcpyDeviceToHost);

    // double maxError=0.0f;
    // for(int i=0;i<n;i++){
    //     double err = hzCheck[i] - hz[i];
    //     maxError = fmax(maxError,err);
    // }
    // std::cout << maxError << std::endl;
    for(int i=0;i<n;i++){
        if(i>0 && i%8==0)
            std::cout << std::endl;
        // std::cout << hz[i] << " ";
        printf("%f ",__half2float(hz[i]));
    }

    free(hx);
    free(hy);
    free(hz);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    return 0;
}
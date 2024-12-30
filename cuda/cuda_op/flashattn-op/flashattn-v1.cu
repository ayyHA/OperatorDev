#include <cuda_runtime.h>
#include <cuda.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <iostream>

__global__ void flash_attn_v1(float* Q,float* K,float* V,
                              int Tr,int Tc,int Br,int Bc,int N,int d,float softmax_scale,
                              float* O,float* l,float* m){
    const int tx = threadIdx.x;
    const int bx = blockIdx.x,by = blockIdx.y;

    const int qkv_offset = (bx*gridDim.y*N*d) + (by*N*d);    // 跳到bx负责的样本,跳到by负责的头
    const int lm_offset = (bx*gridDim.y*N) + (by*N);

    extern __shared__ float smem[];
    int tile_size = Br * d; // 如果Br,Bc不同的话得拆成俩
    float* Qi = smem;
    float* Kj = &smem[tile_size];
    float* Vj = &smem[tile_size*2];
    float* Si = &smem[tile_size*3];

    for(int j=0;j<Tc;j++){
        for(int x=0;x<d;x++){
            // 把此刻的Kj,Vj搬运到smem上
            Kj[tx*d + x] = K[qkv_offset + j*tile_size + tx*d + x];
            Vj[tx*d + x] = V[qkv_offset + j*tile_size + tx*d + x];
        }

        __syncthreads();    // 同步一下，大家看到一样的共享内存

        for(int i=0;i<Tr;i++){
            for(int x=0;x<d;x++){
                Qi[tx*d + x] = Q[qkv_offset + i*tile_size + tx*d + x];
            }            
            float m_prev = m[lm_offset + i*Br + tx];    // 此head_dim上一轮的最大值
            float l_prev = l[lm_offset + i*Br + tx];    // ...累和
            
            float m_now = -INFINITY;
            float l_now = 0;

            // S=QK^T m_new
            for(int y=0;y<Bc;y++){
                float sum = 0;
                for(int x=0;x<d;x++){
                    sum += Qi[tx*d + x] * Kj[y*d + x];
                }
                sum *= softmax_scale;
                if(sum>m_now)
                    m_now = sum;
                Si[tx*Bc + y] = sum;    // Si的shape是Br x Bc
            }
            
            // P=e^(S-m) safe softmax
            for(int y=0;y<Bc;y++){
                Si[tx*Bc + y] = __expf(Si[tx*Bc +y] - m_now);
                l_now += Si[tx*Bc + y];
            }

            float m_new = max(m_prev,m_now);
            float l_new = __expf(m_prev - m_new)*l_prev + __expf(m_now - m_new)*l_now;

            for(int x=0;x<d;x++){
                float pv = 0;
                for(int y=0;y<Bc;y++){
                    pv += Si[tx*Bc + y] * Vj[y*d + x];
                }
                O[qkv_offset + i*tile_size + tx*d+x] = (__expf(m_now - m_new)*pv + l_prev*__expf(m_prev - m_new)*O[qkv_offset + i*tile_size + tx*d + x])/l_new;
            }
            m[lm_offset + i*Br + tx] = m_new;
            l[lm_offset + i*Br + tx] = l_new;
        }
        __syncthreads();
    }
}

torch::Tensor forward(torch::Tensor Q,torch::Tensor K,torch::Tensor V){
    const int Bc = 32;
    const int Br = 32;

    const int b = Q.size(0);   // batch_size
    const int nh = Q.size(1);    // num_heads
    const int N = Q.size(2);   // sequence_length 
    const int d = Q.size(3);     // head_dimension

    const int Tc = ceil(N*1.0/Bc);
    const int Tr = ceil(N*1.0/Br);
    const float softmax_scale = 1.0/sqrt(d);

    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({b,nh,N});
    auto m = torch::full({b,nh,N},-INFINITY);
    torch::Device device(torch::kCUDA); 
    l = l.to(device);
    m = m.to(device);
    int smem_size = 2*Bc*d*sizeof(float) + Br*d*sizeof(float) + Br*Bc*sizeof(float);   // K,V | Q | O或者说是S
    // 需要补充maxSmem的,如果超出,则直接return -1啥的
    
    
    dim3 grid(b,nh);
    dim3 block(Bc);
    flash_attn_v1<<<grid,block,smem_size>>>(
        Q.data_ptr<float>(),K.data_ptr<float>(),V.data_ptr<float>(),
        Tr,Tc,Br,Bc,N,d,softmax_scale,
        O.data_ptr<float>(),l.data_ptr<float>(),m.data_ptr<float>());

    return O;
}

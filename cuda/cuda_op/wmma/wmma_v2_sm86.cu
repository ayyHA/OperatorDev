#include <cstdio>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <climits>
#include <cmath>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define ROW_OFFSET(i,j,ld) ((i)*(ld)+(j))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define CEIL_OPERTAION(x,y) ( ((x)+(y)-1)/(y) )

#define checkCudaError(func) {                                                          \
    cudaError_t e = (func);                                                             \
    if(e !=cudaSuccess){                                                                \
        printf("CUDA ERROR %s %d : %s\n",__FILE__,__LINE__,cudaGetErrorString(e));      \
    }                                                                                   \
}

/*
    1. 比较朴素的GEMM,利用WMMA API进行计算,利用了共享内存减少访问GMEM的次数
 */
// wmma_v2_naive<128,256,32,256><<<grid,block>>>(...)
template<unsigned int BLOCK_SIZE_M,unsigned int BLOCK_SIZE_N,unsigned int BLOCK_SIZE_K, unsigned int THREAD_PER_BLOCK>
__global__ void wmma_v2_naive_aligned(__restrict__ half* A, __restrict__ half* B, 
                        __restrict__ half* C,int M,int N,int K){
    int tid = threadIdx.x;
    int warpId = tid/32;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 1. 把数据从gmem搬到smem上去
    __shared__ half smemA[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ half smemB[BLOCK_SIZE_K][BLOCK_SIZE_N];
    // __shared__ half smemC[BLOCK_SIZE_M][BLOCK_SIZE_N];
    // 每个thread要搬运多少元素
    const int ld_a_elements = BLOCK_SIZE_M * BLOCK_SIZE_K /  THREAD_PER_BLOCK;
    const int ld_b_elements = BLOCK_SIZE_K * BLOCK_SIZE_N /  THREAD_PER_BLOCK;
    // 每个thread要搬运多少次,向量化访存,一次搬运8个fp16
    const int ld_a_times = ld_a_elements/8;
    const int ld_b_times = ld_b_elements/8;
    // 搬运A,B阵一行需要的线程数
    const int a_thread_num_per_line = BLOCK_SIZE_K/8;
    const int b_thread_num_per_line = BLOCK_SIZE_N/8;
    // 每个thread对应搬运到smem的位置
    const int ld_a_smem_m = tid / a_thread_num_per_line;
    const int ld_a_smem_n = (tid % a_thread_num_per_line)* 8;
    const int ld_b_smem_m = tid / b_thread_num_per_line;
    const int ld_b_smem_n = (tid % b_thread_num_per_line) * 8;
    // 所有线程搬运A,B一次的步距(即行数)
    const int ld_a_stride = THREAD_PER_BLOCK / a_thread_num_per_line;
    const int ld_b_stride = THREAD_PER_BLOCK / b_thread_num_per_line;
    // splitk,沿着K方向需要遍历的次数
    const int ktimes = K/BLOCK_SIZE_K;
    
    // A,B起始的位置
    half* tmpA = A + (by*BLOCK_SIZE_M)*K;
    half* tmpB = B + bx*BLOCK_SIZE_N;

    // 1个warp处理64x64的C
    wmma::fragment<wmma::accumulator,WMMA_M,WMMA_N,WMMA_K,half> fragC[4][4];
    // fragA处理64x32的A
    wmma::fragment<wmma::matrix_a,WMMA_M,WMMA_N,WMMA_K,half,wmma::row_major> fragA[2][4];
    // fragB处理32x64的B
    wmma::fragment<wmma::matrix_b,WMMA_M,WMMA_N,WMMA_K,half,wmma::row_major> fragB[2][4];

    // warp对应的m,n用于处理warp level的操作
    const int warp_m = warpId & 1;  // warp % 2
    const int warp_n = warpId >> 1; // warp / 2

    // 初始化C阵
    #pragma unroll
    for(int i=0;i<4;i++){
        #pragma unroll
        for(int j=0;j<4;j++){
            wmma::fill_fragment(fragC[i][j],0.0f);
        }
    }

    // 沿着K维移动,先gmem->smem
    // #pragma unroll
    for(int bk=0;bk<ktimes;bk++){
        #pragma unroll
        for(int i=0;i<ld_a_times;i++){
            int stride = i*ld_a_stride;
            FLOAT4(smemA[ld_a_smem_m + stride][ld_a_smem_n]) = FLOAT4(tmpA[ROW_OFFSET(ld_a_smem_m + stride,ld_a_smem_n,K)]); 
        }

        #pragma unroll
        for(int i=0;i<ld_b_times;i++){
            int stride = i*ld_b_stride;
            FLOAT4(smemB[ld_b_smem_m + stride][ld_b_smem_n]) = FLOAT4(tmpB[ROW_OFFSET(ld_b_smem_m + stride,ld_b_smem_n,N)]);
        }
        // 更新tmpA,tmpB的起始位置
        tmpA = tmpA + BLOCK_SIZE_K;
        tmpB = tmpB + BLOCK_SIZE_K * N;
        __syncthreads();

        // 按照warp来load_matrix smem->reg
        // 对fragA
        wmma::load_matrix_sync(fragA[0][0],&smemA[warp_m*64][0],BLOCK_SIZE_K);     
        wmma::load_matrix_sync(fragA[0][1],&smemA[warp_m*64 + 16][0],BLOCK_SIZE_K);     
        wmma::load_matrix_sync(fragA[0][2],&smemA[warp_m*64 + 32][0],BLOCK_SIZE_K);     
        wmma::load_matrix_sync(fragA[0][3],&smemA[warp_m*64 + 48][0],BLOCK_SIZE_K);     
        wmma::load_matrix_sync(fragA[1][0],&smemA[warp_m*64][16],BLOCK_SIZE_K);     
        wmma::load_matrix_sync(fragA[1][1],&smemA[warp_m*64 + 16][16],BLOCK_SIZE_K);     
        wmma::load_matrix_sync(fragA[1][2],&smemA[warp_m*64 + 32][16],BLOCK_SIZE_K);     
        wmma::load_matrix_sync(fragA[1][3],&smemA[warp_m*64 + 48][16],BLOCK_SIZE_K);     
        // 对fragB
        wmma::load_matrix_sync(fragB[0][0],&smemB[0][warp_n*64],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[0][1],&smemB[0][warp_n*64+16],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[0][2],&smemB[0][warp_n*64+32],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[0][3],&smemB[0][warp_n*64+48],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[1][0],&smemB[16][warp_n*64],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[1][1],&smemB[16][warp_n*64+16],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[1][2],&smemB[16][warp_n*64+32],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[1][3],&smemB[16][warp_n*64+48],BLOCK_SIZE_N);

        // MMA开乘
        #pragma unroll
        for(int i=0;i<4;i++){
            #pragma unroll
            for(int j=0;j<4;j++){
                wmma::mma_sync(fragC[i][j],fragA[0][i],fragB[0][j],fragC[i][j]);
                wmma::mma_sync(fragC[i][j],fragA[1][i],fragB[1][j],fragC[i][j]);
            }
        }

        __syncthreads(); 
    }

    int st_c_gmem_m = by * BLOCK_SIZE_M + warp_m * 64;  // 算出每个warp对应C的行数
    int st_c_gmem_n = bx * BLOCK_SIZE_N + warp_n * 64;  // 算出每个warp对应C的列数
    #pragma unroll
    for(int i=0;i<4;i++){
        #pragma unroll
        for(int j=0;j<4;j++){
            wmma::store_matrix_sync(&C[ROW_OFFSET(
                st_c_gmem_m + i*16,
                st_c_gmem_n + j*16,
                N
            )],fragC[i][j],N,wmma::mem_row_major);
        }
    }
}

/*
    2. 利用异步拷贝直接GMEM->SMEM,不通过寄存器中转
*/

template<unsigned int BLOCK_SIZE_M, unsigned int BLOCK_SIZE_N, unsigned int BLOCK_SIZE_K,unsigned int THREAD_PER_BLOCK>
__global__ void wmma_v3_cpAsync_aligned(__restrict__ half* A, __restrict__ half* B,
                                 __restrict__ half* C, int M, int N, int K){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int warpId = tid >> 5;  // 获取tid对应的warpId

    __shared__ half smemA[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ half smemB[BLOCK_SIZE_K][BLOCK_SIZE_N];
    
    // A,B阵一行所需要的线程数目,向量化访存,fp16相当于1行GET 8个元素
    const int a_thread_num_per_line = BLOCK_SIZE_K / 8;
    const int b_thread_num_per_line = BLOCK_SIZE_N / 8;
    // A,B阵步距,即所有线程1次处理A多少行,B多少行
    const int ld_a_stride = THREAD_PER_BLOCK / a_thread_num_per_line;
    const int ld_b_stride = THREAD_PER_BLOCK / b_thread_num_per_line;
    // 处理A,B阵的次数
    const int ld_a_times = BLOCK_SIZE_M / ld_a_stride;
    const int ld_b_times = BLOCK_SIZE_K / ld_b_stride;
    // 每一个线程取对应的A,B阵的行列
    const int ld_a_smem_m = tid / a_thread_num_per_line;
    const int ld_a_smem_n = (tid % a_thread_num_per_line)<<3;
    const int ld_b_smem_m = tid / b_thread_num_per_line;
    const int ld_b_smem_n = (tid % b_thread_num_per_line)<<3;
    // 获取smemA,smemB的shared state space pointer
    size_t p_smemA_base = __cvta_generic_to_shared(smemA[0]);
    size_t p_smemB_base = __cvta_generic_to_shared(smemB[0]);
    // 当前block内线程处理的gmem的对应行列
    int ld_a_gmem_m = by * BLOCK_SIZE_M + ld_a_smem_m;
    int ld_b_gmem_n = bx * BLOCK_SIZE_N + ld_b_smem_n;
    // 沿着K遍历需要迭代的次数
    int ktimes = K/BLOCK_SIZE_K;

    // 建fragment
    wmma::fragment<wmma::matrix_a,WMMA_M,WMMA_N,WMMA_K,half,wmma::row_major> fragA[2][4];
    wmma::fragment<wmma::matrix_b,WMMA_M,WMMA_N,WMMA_K,half,wmma::row_major> fragB[2][4];
    wmma::fragment<wmma::accumulator,WMMA_M,WMMA_N,WMMA_K,half> fragC[4][4];

    // 初始化fragC:
    #pragma unroll
    for(int i=0;i<4;i++){
        #pragma unroll
        for(int j=0;j<4;j++){
            wmma::fill_fragment(fragC[i][j],0.0f);
        }
    }

    const int warp_m = warpId & 1;
    const int warp_n = warpId >>1;

    // 大循环
    for(int bk=0;bk<ktimes;bk++){
        // 异步加载元素:
        // A gmem->smem
        #pragma unorll
        for(int i=0;i<ld_a_times;i++){
            int stride = i*ld_a_stride;
            asm volatile(
                "cp.async.ca.shared.global [%0],[%1],16;    \n"
                ::"l"(p_smemA_base + ROW_OFFSET(ld_a_smem_m + stride,ld_a_smem_n,BLOCK_SIZE_K)*(int)sizeof(half)),
                  "l"(&A[ROW_OFFSET(ld_a_gmem_m + stride,ld_a_smem_n + bk*BLOCK_SIZE_K,K)])
            );       
        }

        // B gmem->smem
        #pragma unroll
        for(int i=0;i<ld_b_times;i++){
            int stride = i*ld_b_stride;
            asm volatile(
                "cp.async.ca.shared.global [%0],[%1],16;    \n"
                :: "l"(p_smemB_base + ROW_OFFSET(ld_b_smem_m + stride,ld_b_smem_n,BLOCK_SIZE_N)*(int)sizeof(half)),
                   "l"(&B[ROW_OFFSET(ld_b_smem_m + stride + bk*BLOCK_SIZE_K,ld_b_gmem_n,N)])
            );
        }

        asm volatile("cp.async.commit_group;    \n" ::);
        asm volatile("cp.async.wait_group 0;    \n" ::);
        __syncthreads();
        // smemA -> fragA
        wmma::load_matrix_sync(fragA[0][0],&smemA[warp_m*64][0],BLOCK_SIZE_K);
        wmma::load_matrix_sync(fragA[0][1],&smemA[warp_m*64+16][0],BLOCK_SIZE_K);
        wmma::load_matrix_sync(fragA[0][2],&smemA[warp_m*64+32][0],BLOCK_SIZE_K);
        wmma::load_matrix_sync(fragA[0][3],&smemA[warp_m*64+48][0],BLOCK_SIZE_K);
        wmma::load_matrix_sync(fragA[1][0],&smemA[warp_m*64][16],BLOCK_SIZE_K);
        wmma::load_matrix_sync(fragA[1][1],&smemA[warp_m*64+16][16],BLOCK_SIZE_K);
        wmma::load_matrix_sync(fragA[1][2],&smemA[warp_m*64+32][16],BLOCK_SIZE_K);
        wmma::load_matrix_sync(fragA[1][3],&smemA[warp_m*64+48][16],BLOCK_SIZE_K);  
        // smemB -> fragB
        wmma::load_matrix_sync(fragB[0][0],&smemB[0][warp_n*64],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[0][1],&smemB[0][warp_n*64 +16],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[0][2],&smemB[0][warp_n*64 +32],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[0][3],&smemB[0][warp_n*64 +48],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[1][0],&smemB[16][warp_n*64],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[1][1],&smemB[16][warp_n*64 +16],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[1][2],&smemB[16][warp_n*64 +32],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[1][3],&smemB[16][warp_n*64 +48],BLOCK_SIZE_N);

        // 开乘
        #pragma unroll
        for(int i=0;i<4;i++){
            #pragma unroll
            for(int j=0;j<4;j++){
                wmma::mma_sync(fragC[i][j],fragA[0][i],fragB[0][j],fragC[i][j]);
                wmma::mma_sync(fragC[i][j],fragA[1][i],fragB[1][j],fragC[i][j]);
            }
        }

        __syncthreads();    // 万一warp0飞起,异步读取也再一次读完,并提交,但是warp1巨慢,刚好加载fragB,这时就读到的是新的smemB了(部分新),因此这里得同步一下
    }
    
    int st_c_gmem_m = by*BLOCK_SIZE_M + warp_m*64;
    int st_c_gmem_n = bx*BLOCK_SIZE_N + warp_n*64;
    
    #pragma unroll
    for(int i=0;i<4;i++){
        #pragma unroll
        for(int j=0;j<4;j++){
            wmma::store_matrix_sync(&C[ROW_OFFSET(st_c_gmem_m+i*16,st_c_gmem_n+j*16,N)],
                            fragC[i][j],N,wmma::mem_row_major);
        }
    }
}

/*
    3. 双缓冲
*/

template<unsigned int BLOCK_SIZE_M, unsigned int BLOCK_SIZE_N, unsigned int BLOCK_SIZE_K, unsigned int THREAD_PER_BLOCK>
__global__ void wmma_v4_db_aligned(__restrict__ half* A, __restrict__ half* B,
                            __restrict__ half* C,int M,int N,int K){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int warpId = tid >> 5;
    
    // A,B阵每行需要多少个线程;
    const int a_thread_num_per_line = BLOCK_SIZE_K / 8;
    const int b_thread_num_per_line = BLOCK_SIZE_N / 8;
    // 所有线程1次可以处理A,B阵的行数
    const int ld_a_stride = THREAD_PER_BLOCK / a_thread_num_per_line;
    const int ld_b_stride = THREAD_PER_BLOCK / b_thread_num_per_line;   
    // 读取A,B阵需要的次数
    const int ld_a_times = BLOCK_SIZE_M / ld_a_stride;
    const int ld_b_times = BLOCK_SIZE_K / ld_b_stride;
    // 每个线程对应处理A,B阵的具体的行列
    const int ld_a_smem_m = tid / a_thread_num_per_line;
    const int ld_a_smem_n = (tid % a_thread_num_per_line)<<3;
    const int ld_b_smem_m = tid / b_thread_num_per_line;
    const int ld_b_smem_n = (tid % b_thread_num_per_line)<<3;
    // 因为是warp level处理的,所以需要把warp对应的m,n算出来
    const int warp_m = warpId % 2;  // 偶数第0行,奇数第1行
    const int warp_n = warpId / 2;
    // 动态共享内存
    extern __shared__ half smem[];
    // 含双缓冲,指向各自对应的空间开始处
    half* smemA = smem;
    half* smemB = smem + 2*(BLOCK_SIZE_M * BLOCK_SIZE_K);
    // 双缓冲的stride,这是元素个数
    const int ld_a_buffer_stride = BLOCK_SIZE_M * BLOCK_SIZE_K;
    const int ld_b_buffer_stride = BLOCK_SIZE_K * BLOCK_SIZE_N;
    // fragment,1个warp处理64*32个数据,对于A,B阵而言,而1个fragment存储的是16x16的片段信息,因此需要多个fragment
    wmma::fragment<wmma::matrix_a,WMMA_M,WMMA_N,WMMA_K,half,wmma::row_major> fragA[2][4];
    wmma::fragment<wmma::matrix_b,WMMA_M,WMMA_N,WMMA_K,half,wmma::row_major> fragB[2][4];
    wmma::fragment<wmma::accumulator,WMMA_M,WMMA_N,WMMA_K,half> fragC[4][4];

    // 初始fragC
    #pragma unroll
    for(int i=0;i<4;i++){
        #pragma unroll
        for(int j=0;j<4;j++){
            wmma::fill_fragment(fragC[i][j],0.0f);
        }
    }

    // 每个线程对应到原A阵的行数和原B阵的列数
    int ld_a_gmem_m = by * BLOCK_SIZE_M + ld_a_smem_m;
    int ld_b_gmem_n = bx * BLOCK_SIZE_N + ld_b_smem_n;

    int ld_a_gmem_addr = ROW_OFFSET(ld_a_gmem_m,ld_a_smem_n,K);
    int ld_b_gmem_addr = ROW_OFFSET(ld_b_smem_m,ld_b_gmem_n,N);

    // 因为用了PTX指令,所以要指定共享内存地址的SS
    size_t ptr_a_smem_base = __cvta_generic_to_shared(smemA);
    size_t ptr_b_smem_base = __cvta_generic_to_shared(smemB);

    // 异步拷贝,双缓冲,先拷一点
    {   
        #pragma unroll
        for(int i=0;i<ld_a_times;i++){
            int stride = i*ld_a_stride;
            asm volatile("cp.async.ca.shared.global [%0],[%1],16;   \n"
            :: "l"(ptr_a_smem_base + ROW_OFFSET(ld_a_smem_m + stride,ld_a_smem_n,BLOCK_SIZE_K)*(int)sizeof(half)),
               "l"(&A[ld_a_gmem_addr + stride*K]));
        }   

        #pragma unroll
        for(int i=0;i<ld_b_times;i++){
            int stride = i*ld_b_stride;
            asm volatile("cp.async.ca.shared.global [%0],[%1],16;   \n"
            :: "l"(ptr_b_smem_base + ROW_OFFSET(ld_b_smem_m + stride,ld_b_smem_n,BLOCK_SIZE_N)*(int)sizeof(half)),
               "l"(&B[ld_b_gmem_addr + stride*N]));
        }

        asm volatile("cp.async.commit_group;    \n" ::);
        asm volatile("cp.async.wait_group 0;    \n" ::);
        __syncthreads();
    }

    int ktimes = K / BLOCK_SIZE_K;
    int read_idx = 0;
    // 读了ktimes次,但计算少了1次,外面补上
    for(int bk=1;bk<ktimes;bk++){
        int write_idx = read_idx^1;
        // 更新取gmem的地址,往下一个bk移动
        ld_a_gmem_addr += BLOCK_SIZE_K;
        ld_b_gmem_addr += BLOCK_SIZE_K * N;

        #pragma unroll
        for(int i=0;i<ld_a_times;i++){
            int stride = i*ld_a_stride;
            asm volatile("cp.async.ca.shared.global [%0],[%1],16;   \n"
            :: "l"(ptr_a_smem_base + (write_idx * ld_a_buffer_stride + ROW_OFFSET(ld_a_smem_m + stride,ld_a_smem_n,BLOCK_SIZE_K)) * (int)sizeof(half)),
               "l"(&A[ld_a_gmem_addr + stride*K]));
        }   

        #pragma unroll
        for(int i=0;i<ld_b_times;i++){
            int stride = i*ld_b_stride;
            asm volatile("cp.async.ca.shared.global [%0],[%1],16;   \n"
            :: "l"(ptr_b_smem_base + (write_idx * ld_b_buffer_stride + ROW_OFFSET(ld_b_smem_m + stride,ld_b_smem_n,BLOCK_SIZE_N)) * (int)sizeof(half)),
               "l"(&B[ld_b_gmem_addr + stride*N]));
        }

        // 加载数据smem->reg
        wmma::load_matrix_sync(fragA[0][0],&smemA[ROW_OFFSET(warp_m*64 +    read_idx*BLOCK_SIZE_M   ,0,BLOCK_SIZE_K)]      ,BLOCK_SIZE_K);
        wmma::load_matrix_sync(fragA[0][1],&smemA[ROW_OFFSET(warp_m*64+16 + read_idx*BLOCK_SIZE_M   ,0,BLOCK_SIZE_K)]      ,BLOCK_SIZE_K);
        wmma::load_matrix_sync(fragA[0][2],&smemA[ROW_OFFSET(warp_m*64+32 + read_idx*BLOCK_SIZE_M   ,0,BLOCK_SIZE_K)]      ,BLOCK_SIZE_K);
        wmma::load_matrix_sync(fragA[0][3],&smemA[ROW_OFFSET(warp_m*64+48 + read_idx*BLOCK_SIZE_M   ,0,BLOCK_SIZE_K)]      ,BLOCK_SIZE_K);
        wmma::load_matrix_sync(fragA[1][0],&smemA[ROW_OFFSET(warp_m*64 +    read_idx*BLOCK_SIZE_M   ,16,BLOCK_SIZE_K)]     ,BLOCK_SIZE_K);
        wmma::load_matrix_sync(fragA[1][1],&smemA[ROW_OFFSET(warp_m*64+16 + read_idx*BLOCK_SIZE_M   ,16,BLOCK_SIZE_K)]     ,BLOCK_SIZE_K);
        wmma::load_matrix_sync(fragA[1][2],&smemA[ROW_OFFSET(warp_m*64+32 + read_idx*BLOCK_SIZE_M   ,16,BLOCK_SIZE_K)]     ,BLOCK_SIZE_K);
        wmma::load_matrix_sync(fragA[1][3],&smemA[ROW_OFFSET(warp_m*64+48 + read_idx*BLOCK_SIZE_M   ,16,BLOCK_SIZE_K)]     ,BLOCK_SIZE_K);
        
        wmma::load_matrix_sync(fragB[0][0],&smemB[ROW_OFFSET(read_idx*BLOCK_SIZE_K,         warp_n*64,      BLOCK_SIZE_N)],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[0][1],&smemB[ROW_OFFSET(read_idx*BLOCK_SIZE_K,         warp_n*64+16,   BLOCK_SIZE_N)],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[0][2],&smemB[ROW_OFFSET(read_idx*BLOCK_SIZE_K,         warp_n*64+32,   BLOCK_SIZE_N)],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[0][3],&smemB[ROW_OFFSET(read_idx*BLOCK_SIZE_K,         warp_n*64+48,   BLOCK_SIZE_N)],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[1][0],&smemB[ROW_OFFSET(16 + read_idx*BLOCK_SIZE_K,    warp_n*64,      BLOCK_SIZE_N)],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[1][1],&smemB[ROW_OFFSET(16 + read_idx*BLOCK_SIZE_K,    warp_n*64+16,   BLOCK_SIZE_N)],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[1][2],&smemB[ROW_OFFSET(16 + read_idx*BLOCK_SIZE_K,    warp_n*64+32,   BLOCK_SIZE_N)],BLOCK_SIZE_N);
        wmma::load_matrix_sync(fragB[1][3],&smemB[ROW_OFFSET(16 + read_idx*BLOCK_SIZE_K,    warp_n*64+48,   BLOCK_SIZE_N)],BLOCK_SIZE_N);
        // mma开乘
        #pragma unroll
        for(int i=0;i<4;i++){
            #pragma unroll
            for(int j=0;j<4;j++){
                wmma::mma_sync(fragC[i][j],fragA[0][i],fragB[0][j],fragC[i][j]);
                wmma::mma_sync(fragC[i][j],fragA[1][i],fragB[1][j],fragC[i][j]);
            }
        }
        // 算完更新read_idx
        read_idx ^=1;
        // 同步一下,异步拷贝的是下一次要的,不急着要,这里一起同步了就好,上面省去一个同步的
        asm volatile("cp.async.commit_group;    \n" ::);
        asm volatile("cp.async.wait_group 0;    \n" ::);
        __syncthreads();
    }
    // smem->reg
    wmma::load_matrix_sync(fragA[0][0],&smemA[ROW_OFFSET(warp_m*64 + read_idx*BLOCK_SIZE_M,     0,BLOCK_SIZE_K)],       BLOCK_SIZE_K);
    wmma::load_matrix_sync(fragA[0][1],&smemA[ROW_OFFSET(warp_m*64+16 + read_idx*BLOCK_SIZE_M,  0,BLOCK_SIZE_K)],       BLOCK_SIZE_K);
    wmma::load_matrix_sync(fragA[0][2],&smemA[ROW_OFFSET(warp_m*64+32 + read_idx*BLOCK_SIZE_M,  0,BLOCK_SIZE_K)],       BLOCK_SIZE_K);
    wmma::load_matrix_sync(fragA[0][3],&smemA[ROW_OFFSET(warp_m*64+48 + read_idx*BLOCK_SIZE_M,  0,BLOCK_SIZE_K)],       BLOCK_SIZE_K);
    wmma::load_matrix_sync(fragA[1][0],&smemA[ROW_OFFSET(warp_m*64 + read_idx*BLOCK_SIZE_M,     16,BLOCK_SIZE_K)],      BLOCK_SIZE_K);
    wmma::load_matrix_sync(fragA[1][1],&smemA[ROW_OFFSET(warp_m*64+16 + read_idx*BLOCK_SIZE_M,  16,BLOCK_SIZE_K)],      BLOCK_SIZE_K);
    wmma::load_matrix_sync(fragA[1][2],&smemA[ROW_OFFSET(warp_m*64+32 + read_idx*BLOCK_SIZE_M,  16,BLOCK_SIZE_K)],      BLOCK_SIZE_K);
    wmma::load_matrix_sync(fragA[1][3],&smemA[ROW_OFFSET(warp_m*64+48 + read_idx*BLOCK_SIZE_M,  16,BLOCK_SIZE_K)],      BLOCK_SIZE_K);
    
    wmma::load_matrix_sync(fragB[0][0],&smemB[ROW_OFFSET(read_idx*BLOCK_SIZE_K,         warp_n*64,      BLOCK_SIZE_N)],BLOCK_SIZE_N);
    wmma::load_matrix_sync(fragB[0][1],&smemB[ROW_OFFSET(read_idx*BLOCK_SIZE_K,         warp_n*64+16,   BLOCK_SIZE_N)],BLOCK_SIZE_N);
    wmma::load_matrix_sync(fragB[0][2],&smemB[ROW_OFFSET(read_idx*BLOCK_SIZE_K,         warp_n*64+32,   BLOCK_SIZE_N)],BLOCK_SIZE_N);
    wmma::load_matrix_sync(fragB[0][3],&smemB[ROW_OFFSET(read_idx*BLOCK_SIZE_K,         warp_n*64+48,   BLOCK_SIZE_N)],BLOCK_SIZE_N);
    wmma::load_matrix_sync(fragB[1][0],&smemB[ROW_OFFSET(16 + read_idx*BLOCK_SIZE_K,    warp_n*64,      BLOCK_SIZE_N)],BLOCK_SIZE_N);
    wmma::load_matrix_sync(fragB[1][1],&smemB[ROW_OFFSET(16 + read_idx*BLOCK_SIZE_K,    warp_n*64+16,   BLOCK_SIZE_N)],BLOCK_SIZE_N);
    wmma::load_matrix_sync(fragB[1][2],&smemB[ROW_OFFSET(16 + read_idx*BLOCK_SIZE_K,    warp_n*64+32,   BLOCK_SIZE_N)],BLOCK_SIZE_N);
    wmma::load_matrix_sync(fragB[1][3],&smemB[ROW_OFFSET(16 + read_idx*BLOCK_SIZE_K,    warp_n*64+48,   BLOCK_SIZE_N)],BLOCK_SIZE_N);
    // 漏掉的最后一次计算
    #pragma unroll
    for(int i=0;i<4;i++){
        #pragma unroll
        for(int j=0;j<4;j++){
            wmma::mma_sync(fragC[i][j],fragA[0][i],fragB[0][j],fragC[i][j]);
            wmma::mma_sync(fragC[i][j],fragA[1][i],fragB[1][j],fragC[i][j]);
        }
    }

    // 存起来,依旧逐warp,需要注意
    const int st_c_gmem_m = by * BLOCK_SIZE_M + warp_m * 64;
    const int st_c_gmem_n = bx * BLOCK_SIZE_N + warp_n * 64;

    #pragma unroll
    for(int i=0;i<4;i++){
        #pragma unroll
        for(int j=0;j<4;j++){
            wmma::store_matrix_sync(&C[ROW_OFFSET(st_c_gmem_m+i*16,st_c_gmem_n+j*16,N)],fragC[i][j],N,wmma::mem_row_major);
        }
    }
}
/* 
    4.考虑L2 Cache的局部性,因为对BLOCK落到SM的调度是(GRID)Z->Y->X,通过执行配置参数增加Z维,使得L2 Cache的局部性利用起来
    不然按照MNK->16384,16384,16384这个size来说,1个SM只能调度1个block,按照BM=128,BN=256则是有128x64个block
    按照RTX3090共82个SM考虑:
      如果按照grid(64,128):第一行排满,第二行排18个即可,这样对应到A阵是局部性很高,但是对应到B阵局部性一塌糊涂;
      如果按照grid(16,128,4):排满前五行,第六行放两个block,如此调度~
    显然后者更优,A,B阵的局部性利用得很均衡
*/
// __global__ void wmma_v5_l2cache_aligned(){

// }
// /*
//     5. 大循环给它unroll
//  */
// __global__ void wmma_v6_unroll_aligned(){

// }

void cpuHGEMM(half* A,half* B,half* C,int M,int N,int K){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            float psum = 0.0f;
            for(int k=0;k<K;k++){
                psum += (float)A[ROW_OFFSET(i,k,K)] * (float)B[ROW_OFFSET(k,j,N)];
            }
        C[ROW_OFFSET(i,j,N)] = (half)psum;
        }
    }
}

void testError(void (*wmma_func)(half*,half*,half*,int,int,int),
            int M,int N,int K){
    size_t sizeA = M*K*sizeof(half);
    size_t sizeB = K*N*sizeof(half);
    size_t sizeC = M*N*sizeof(half);
    half* hA,*hB,*hC,*hdC;
    hA = (half*)malloc(sizeA);
    hB = (half*)malloc(sizeB);
    hC = (half*)malloc(sizeC);
    hdC = (half*)malloc(sizeC);
    for(int i=0;i<M*K;i++){
        hA[i] = (half)drand48();
    }
    for(int i=0;i<K*N;i++){
        hB[i] = (half)drand48();
    }

    half* dA,*dB,*dC;
    checkCudaError(cudaMalloc(&dA,sizeA));
    checkCudaError(cudaMalloc(&dB,sizeB));
    checkCudaError(cudaMalloc(&dC,sizeC));

    checkCudaError(cudaMemcpy(dA,hA,sizeA,cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dB,hB,sizeB,cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dC,hC,sizeC,cudaMemcpyHostToDevice));

    // CPU
    cpuHGEMM(hA,hB,hC,M,N,K);

    // GPU
    dim3 block(256);
    dim3 grid(CEIL_OPERTAION(N,256),CEIL_OPERTAION(M,128));

    cudaFuncSetAttribute(wmma_func,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    size_t smemSize = 2*(128*32 + 32*256)*sizeof(half);
    wmma_func<<<grid,block,smemSize>>>(dA,dB,dC,M,N,K);
    checkCudaError(cudaMemcpy(hdC,dC,sizeC,cudaMemcpyDeviceToHost));

    float maxError = 0.0f;
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            float thisError = fabs((float)hC[ROW_OFFSET(i,j,N)] - (float)hdC[ROW_OFFSET(i,j,N)]);
            if(maxError != maxError || thisError != thisError)
                maxError = NAN;
            else
                maxError = max(maxError,thisError); 
        }
    }

    printf("MAXERROR: %.6f\n",maxError);

    free(hA);
    free(hB);
    free(hC);
    checkCudaError(cudaFree(dA));
    checkCudaError(cudaFree(dB));
    checkCudaError(cudaFree(dC));
}

void testPerformance();


int main(int argc,char** argv){
    const int BM = 128;
    const int BN = 256;
    const int BK = 32;
    const int THREAD_PER_BLOCK = 256;

    void (*wmma_v2)(half*,half*,half*,int,int,int) = wmma_v3_cpAsync_aligned<BM,BN,BK,THREAD_PER_BLOCK>;

    testError(wmma_v2,256,256,256);
    return 0;
}
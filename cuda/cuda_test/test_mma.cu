#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>
using std::cout;
using std::endl;
// MNK 16816
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define LDMATRIX_X1(R0, ADDR)                                                  \
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0},[%1];    \n"     \
               : "=r"(R0)                                                      \
               : "r"(ADDR))

#define LDMATRIX_X2(R0, R1, ADDR)                                              \
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2]; \n"     \
               : "=r"(R0), "=r"(R1)                                            \
               : "r"(ADDR))

#define LDMATRIX_X2_T(R0, R1, ADDR)                                            \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16.trans {%0,%1},[%2]; \n"        \
      : "=r"(R0), "=r"(R1)                                                     \
      : "r"(ADDR))

#define LDMATRIX_X4(R0, R1, R2, R3, ADDR)                                      \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4]; \n"        \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(ADDR))

#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)            \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "            \
               "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"                      \
               : "=r"(RD0), "=r"(RD1)                                          \
               : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1),   \
                 "r"(RC0), "r"(RC1))

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])

#define checkCudaError(func)                                                   \
  {                                                                            \
    cudaError_t e = func;                                                      \
    if (e != cudaSuccess) {                                                    \
      cout << __FILE__ << " " << __LINE__                                      \
           << " ERROR:" << cudaGetErrorString(e) << endl;                      \
      exit(-1);                                                                \
    }                                                                          \
  }

template <const int M = 16, const int N = 8, const int K = 16>
__global__ void testHMMA(half *dA, half *dB, half *dC) {
  int tid = threadIdx.x;
  int idx = threadIdx.x * 8;
  __shared__ half sA[M * K];
  __shared__ half sB[K * N];
  __shared__ half sC[M * N];

  constexpr int ld_g2s_a_per_row = K / 8;
  const int ld_g2s_a_m = tid / ld_g2s_a_per_row;
  const int ld_g2s_a_n = (tid % ld_g2s_a_per_row) * 8;
  constexpr int ld_g2s_b_per_row = N / 8;
  const int ld_g2s_b_m = tid / ld_g2s_b_per_row;
  const int ld_g2s_b_n = 0;
  // g2s
  FLOAT4(sA[ld_g2s_a_m * 16 + ld_g2s_a_n]) =
      FLOAT4(dA[ld_g2s_a_m * 16 + ld_g2s_a_n]);
  if (tid < 16)
    FLOAT4(sB[ld_g2s_b_m * 8]) = FLOAT4(dB[ld_g2s_b_m * 8]);
  __syncthreads();
  // s2r
  // 这里计算遵循ldmatrix排列顺序
  const int ldsm_a_m = tid % 16;
  const int ldsm_a_n = (tid / 16) * 8;
  uint32_t smem_a_ptr = __cvta_generic_to_shared(&sA[ldsm_a_m * 16 + ldsm_a_n]);
  const int ldsm_b_m = tid % 16;
  const int ldsm_b_n = 0;
  uint32_t smem_b_ptr = __cvta_generic_to_shared(&sB[ldsm_b_m * 8]);

  uint32_t RA[4];
  uint32_t RB[2];
  uint32_t RC[2] = {0, 0};
  // ldsm
  LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], smem_a_ptr);
  LDMATRIX_X2_T(RB[0], RB[1], smem_b_ptr);
  // hmma
  HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0],
            RC[1]);

  // r2s
  // C:16x8(16x4 u32) row0 T0[r0]:u32 -> half*2 {c0,c1}
  const int st_r2s_c_m = tid / 4;
  const int st_r2s_c_n = (tid % 4) * 2;
  LDST32BITS(sC[st_r2s_c_m * 8 + st_r2s_c_n]) = LDST32BITS(RC[0]);
  LDST32BITS(sC[st_r2s_c_m * 8 + st_r2s_c_n + 64]) = LDST32BITS(RC[1]);
  __syncthreads();
  // s2g
  // 合理来说这些个s2g,g2s都要有gmem,smem各两个单独的坐标,但是我这个例子它可以合起来,就没分开写
  const int st_s2g_c_m = tid;
  const int st_s2g_c_n = 0;
  if (tid < 16) {
    FLOAT4(dC[st_s2g_c_m * 8]) = FLOAT4(sC[st_s2g_c_m * 8]);
  }
}

int main() {
  const int M = 16, N = 8, K = 16;
  half *hA, *hB, *hC;
  half *dA, *dB, *dC;
  hA = (half *)malloc(sizeof(half) * M * K);
  hB = (half *)malloc(sizeof(half) * K * N);
  hC = (half *)malloc(sizeof(half) * M * N);

  float cnt;

  for (int i = 0; i < 16; i++) {
    cnt = .0f;
    for (int j = 0; j < 16; j++) {
      hA[i * 16 + j] = __float2half(cnt++);
      if (cnt == 8.0f)
        cnt = .0f;
    }
    cnt = .0f;
    for (int j = 0; j < 8; j++) {
      hB[i * 8 + j] = __float2half(cnt++);
      if (cnt == 8.0f)
        cnt = .0f;
    }
  }

  checkCudaError(cudaMalloc(&dA, sizeof(half) * M * K));
  checkCudaError(cudaMalloc(&dB, sizeof(half) * N * K));
  checkCudaError(cudaMalloc(&dC, sizeof(half) * M * N));

  checkCudaError(
      cudaMemcpy(dA, hA, sizeof(half) * M * K, cudaMemcpyHostToDevice));
  checkCudaError(
      cudaMemcpy(dB, hB, sizeof(half) * K * N, cudaMemcpyHostToDevice));
  checkCudaError(cudaMemset(dC, 0, sizeof(half) * M * N));
  
  // KERNEL FUNCTION
  testHMMA<<<1, 32>>>(dA, dB, dC);

  checkCudaError(
      cudaMemcpy(hC, dC, sizeof(half) * M * N, cudaMemcpyDeviceToHost));

  cout << endl << "matrixA:" << endl;
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      cout << __half2float(hA[i * 16 + j]) << " ";
    }
    cout << endl;
  }

  cout << endl << "matrixB:" << endl;
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 8; j++) {
      cout << __half2float(hA[i * 8 + j]) << " ";
    }
    cout << endl;
  }

  cout << endl << "RESULT:" << endl;
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 8; j++) {
      cout << __half2float(hC[i * 8 + j]) << " ";
    }
    cout << endl;
  }

  free(hA);
  free(hB);
  free(hC);
  checkCudaError(cudaFree(dA));
  checkCudaError(cudaFree(dB));
  checkCudaError(cudaFree(dC));
  return 0;
}

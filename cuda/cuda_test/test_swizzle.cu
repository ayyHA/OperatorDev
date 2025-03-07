#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

#define WARP_SIZE 32
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

#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

#define checkCudaError(func)                                                   \
  {                                                                            \
    cudaError_t e = func;                                                      \
    if (e != cudaSuccess) {                                                    \
      printf("%s %d, ERROR: %s", __FILE__, __LINE__, cudaGetErrorString(e));   \
      exit(-1);                                                                \
    }                                                                          \
  }
/* g2s s2r(ldsm) r2s(sts) s2g */
template <const int NUM_THREADS = 128, const int M = 16, const int N = 64>
__global__ void naive_copy(half *dA, half *dB) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  const int warpId = tid / WARP_SIZE;
  const int laneId = tid % WARP_SIZE;

  __shared__ half sA[M * N];

  // g2s
  constexpr int a_per_row_threads = N / 8;
  constexpr int a_stride = NUM_THREADS / a_per_row_threads;
  int ld_smem_a_m = tid / a_per_row_threads;
  int ld_smem_a_n = (tid % a_per_row_threads) * 8;
  // just data movement,same coordinate
  int ld_gmem_a_m = ld_smem_a_m;
  int ld_gmem_a_n = ld_smem_a_n;
  LDST128BITS(sA[ld_smem_a_m * N + ld_smem_a_n]) =
      LDST128BITS(dA[ld_gmem_a_m * N + ld_gmem_a_n]);
  __syncthreads();
  // s2r
  int ldsm_a_row = laneId % 16;
  int ldsm_a_col = warpId * 16 + (laneId / 16) * 8;
  uint32_t ldsm_a_ptr =
      __cvta_generic_to_shared(&sA[ldsm_a_row * N + ldsm_a_col]);
  uint32_t RA[4];
  LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], ldsm_a_ptr);
  // r2s put back inplace
  int sts_a_row = laneId / 4;
  int sts_a_col = (laneId % 4) * 2 + warpId * 16;
  LDST32BITS(sA[sts_a_row * N + sts_a_col]) = LDST32BITS(RA[0]);
  LDST32BITS(sA[(sts_a_row + 8) * N + sts_a_col]) = LDST32BITS(RA[1]);
  LDST32BITS(sA[sts_a_row * N + sts_a_col + 8]) = LDST32BITS(RA[2]);
  LDST32BITS(sA[(sts_a_row + 8) * N + sts_a_col + 8]) = LDST32BITS(RA[3]);

  int stg_a_row = laneId % 16;
  int stg_a_col = (laneId / 16) * 8 + warpId * 16;

  // s2g
  // store to matrixB, check copy correction
  int stg_b_row = stg_a_row;
  int stg_b_col = stg_a_col;
  LDST128BITS(dB[stg_b_row * N + stg_b_col]) =
      LDST128BITS(sA[stg_a_row * N + stg_a_col]);
}

/* Only Swizzle ldg and ldsm, we can compare this instructions' bank conflict */
/* icol = irow ^ icol */
template <const int NUM_THREADS = 128, const int M = 16, const int N = 64>
__global__ void swizzle_copy(half *dA, half *dB) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  // printf("bx=%d,by=%d\n",blockIdx.x,blockIdx.y);
  const int laneId = tid % WARP_SIZE;
  const int warpId = tid / WARP_SIZE;

  __shared__ half sA[M * N];

  // g2s
  constexpr int a_per_row_threads = N / 8;
  constexpr int a_stride = NUM_THREADS / a_per_row_threads;
  int ld_a_smem_m = tid / a_per_row_threads;
  int ld_a_smem_n = (tid % a_per_row_threads) * 8;
  int ld_a_gmem_m = ld_a_smem_m;
  int ld_a_gmem_n = ld_a_smem_n;
  // xor
  int ld_a_smem_n_xor = (ld_a_gmem_m & 0x7) ^ (ld_a_gmem_n / 8);
  int ld_a_gmem_addr = ld_a_gmem_m * N + ld_a_gmem_n;
  int ld_a_smem_addr = ld_a_smem_m * N + ld_a_smem_n_xor * 8;
  LDST128BITS(sA[ld_a_smem_addr]) = LDST128BITS(dA[ld_a_gmem_addr]);
  __syncthreads();

  // s2r(ldsm)
  uint32_t RA[4];
  int ldsm_a_m = laneId % 16;
  int ldsm_a_n_xor = (ldsm_a_m & 0x7) ^ (warpId * 2 + laneId / 16);
  uint32_t ldsm_a_ptr = __cvta_generic_to_shared(&sA[ldsm_a_m * N + ldsm_a_n_xor * 8]);
  LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], ldsm_a_ptr);
  // r2s
  int sts_a_row = laneId / 4;
  int sts_a_col = (laneId % 4) * 2 + warpId * 16;
  LDST32BITS(sA[sts_a_row * N + sts_a_col]) = LDST32BITS(RA[0]);
  LDST32BITS(sA[(sts_a_row + 8) * N + sts_a_col]) = LDST32BITS(RA[1]);
  LDST32BITS(sA[sts_a_row * N + sts_a_col + 8]) = LDST32BITS(RA[2]);
  LDST32BITS(sA[(sts_a_row + 8) * N + sts_a_col + 8]) = LDST32BITS(RA[3]);

  int stg_a_row = laneId % 16;
  int stg_a_col = (laneId / 16) * 8 + warpId * 16;

  // s2g
  // store to matrixB, check copy correction
  int stg_b_row = stg_a_row;
  int stg_b_col = stg_a_col;
  LDST128BITS(dB[stg_b_row * N + stg_b_col]) =
      LDST128BITS(sA[stg_a_row * N + stg_a_col]);
}

/*
  Test swizzle with matrix shape 16x64.
  This example only consider load from smem and store to smem.
  Discuss why have bank conflicts and its solution that how to achieve bank
  conflict free.
*/
int main() {
  const int M = 16;
  const int N = 64;
  size_t sizeA = M * N * sizeof(half);
  half *hA = (half *)malloc(sizeA);
  half *hB = (half *)malloc(sizeA);
  for (int i = 0; i < M * N; i++)
    hA[i] = __float2half(float(i));

  half *dA, *dB;
  checkCudaError(cudaMalloc(&dA, sizeA));
  checkCudaError(cudaMalloc(&dB, sizeA));
  checkCudaError(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
  dim3 grid(1);
  dim3 block(32, 4);

  // Test Naive copy
  // naive_copy<<<grid, block>>>(dA, dB);
  // Test Swizzle copy
  swizzle_copy<<<grid, block>>>(dA, dB);
  // Test copy's correction
  checkCudaError(cudaMemcpy(hB, dB, sizeA, cudaMemcpyDeviceToHost));
  double maxError = 0.0f;
  for (int i = 0; i < M * N; i++) {
    float a = __half2float(hA[i]);
    float b = __half2float(hB[i]);
    if (fabs(a - b) > 1e-8) {
      maxError = fabs(a - b);
      printf("ERROR COPY %f,%f\n", a, b);
      exit(-1);
    }
  }
  printf("SUCCESS COPY\n");
  checkCudaError(cudaFree(dA));
  checkCudaError(cudaFree(dB));
  free(hA);
  free(hB);
  return 0;
}
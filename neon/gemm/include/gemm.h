#ifndef _GEMM_H_
#define _GEMM_H_
void gemm_origin(int,int,int,float*,int,float*,int,float*,int);
void gemm_4x1_origin(int,int,int,float*,int,float*,int,float*,int);
void gemm_4x1_reg(int,int,int,float*,int,float*,int,float*,int);
void gemm_4x4_reg(int,int,int,float*,int,float*,int,float*,int);
void gemm_4x4_neon(int,int,int,float*,int,float*,int,float*,int);
void gemm_4x4_neon_block(int,int,int,float*,int,float*,int,float*,int);
void gemm_4x4_neon_packed(int,int,int,float*,int,float*,int,float*,int);
void gemm_4x4_like_blas_neon(int,int,int,float*,int,float*,int,float*,int);
void gemm_4x4_like_blas_asmV1(int,int,int,float*,int,float*,int,float*,int);
void gemm_4x4_like_blas_asmV2(int,int,int,float*,int,float*,int,float*,int);
void gemm_4x4_like_blas_asmV3(int,int,int,float*,int,float*,int,float*,int);
void gemm_8x8_like_blas_neon(int,int,int,float*,int,float*,int,float*,int);

#endif  // _GEMM_H_
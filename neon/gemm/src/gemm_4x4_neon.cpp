#include "gemm.h"
#include "config.h"
#include <algorithm>
#ifdef USE_NEON
    #include <arm_neon.h>
#endif // USE_NEON

#define A(i,j) a[(i)*(lda) + (j)]
#define B(i,j) b[(i)*(ldb) + (j)]
#define C(i,j) c[(i)*(ldc) + (j)]

namespace neon_4x4{
    void add_dot4x4(int,float*,int,float*,int,float*,int);
    void add_dot4x4_unroll4(int,float*,int,float*,int,float*,int);
}

void gemm_4x4_neon(int m,int n,int k,float* a,int lda,
                    float* b,int ldb,float* c,int ldc){
    int i,j;
    for(i=0;i<m;i+=4){
        for(j=0;j<n;j+=4){
            // neon_4x4::add_dot4x4(k,&A(i,0),lda,&B(0,j),ldb,&C(i,j),ldc);
            neon_4x4::add_dot4x4_unroll4(k,&A(i,0),lda,&B(0,j),ldb,&C(i,j),ldc);
        }
    }
}

#define mc 256
#define kc 128
/*
    上面是针对一个矩阵划分成4x4的小块,利用neon intrinsic进行计算(bbuf说是按照"z轴"分块)
    而对于一个很大的A,B阵,实际应该对它们先分块,然后再调用上面的函数进行4x4小块的计算(bbuf说是按照"行列"再进行分块)
*/
void gemm_4x4_neon_block(int m,int n,int k,float* a,int lda,float* b,int ldb,float*c,int ldc){
    int i,p,pb,ib;
    for(i=0;i<m;i+=mc){
        ib = std::min(mc,m-i);
        for(p=0;p<k;p+=kc){
            pb = std::min(kc,k-p);
            gemm_4x4_neon(ib,n,pb,&A(i,p),lda,&B(p,0),ldb,&C(i,0),ldc);
        }
    }
}


namespace neon_4x4{
    void add_dot4x4(int k,float* a,int lda,float* b,int ldb,float* c,int ldc){
        // register float reg_a_0p,reg_a_1p_,reg_a_2p,reg_a_3p;
        float* pnt_a_0p = &A(0,0), *pnt_a_1p = &A(1,0), *pnt_a_2p = &A(2,0), *pnt_a_3p = &A(3,0);
        // float32x4_t C0 = {0},C1 = {0},C2 = {0},C3 = {0};
        float* c00 = &C(0,0);
        float32x4_t C0 = vld1q_f32(c00),C1 = vld1q_f32(c00+ldc), C2 = vld1q_f32(c00+2*ldc), C3 = vld1q_f32(c00+3*ldc);
        float32x4_t B;
        int p;
        for(p=0;p<k;p++){
            B = vld1q_f32(&B(p,0));
            C0 = vfmaq_n_f32(C0,B,*pnt_a_0p++);
            C1 = vfmaq_n_f32(C1,B,*pnt_a_1p++);
            C2 = vfmaq_n_f32(C2,B,*pnt_a_2p++);
            C3 = vfmaq_n_f32(C3,B,*pnt_a_3p++);
        }
        vst1q_f32(c00,C0);
        vst1q_f32(c00+ldc,C1);
        vst1q_f32(c00+2*ldc,C2);
        vst1q_f32(c00+3*ldc,C3);        
    }
    /*
        这个版本考虑了数据的排布方式从而进行展开,但因为白牛大佬他们的做法是上面那种,之后的函数调用都调用类似上面的方式
    */
    void add_dot4x4_unroll4(int k,float* a,int lda,float* b,int ldb,float* c,int ldc){
        // 表示行
        float32x4_t A0,A1,A2,A3;
        float32x4_t B;
        float32x4_t C0,C1,C2,C3;
        
        // float* pnt_b = &B(0,0);
        float*pnt_a0 = &A(0,0),*pnt_a1 = &A(1,0),*pnt_a2 = &A(2,0),*pnt_a3 = &A(3,0);

        C0 = vld1q_f32(&C(0,0));
        C1 = vld1q_f32(&C(1,0));
        C2 = vld1q_f32(&C(2,0));
        C3 = vld1q_f32(&C(3,0));

        int p;
        for(p=0;p<k;p+=4){
            B = vld1q_f32(&B(p,0));
            A0 = vld1q_f32(pnt_a0);
            C0 = vfmaq_laneq_f32(C0,B,A0,0);
            A1 = vld1q_f32(pnt_a1);
            C1 = vfmaq_laneq_f32(C1,B,A1,0);
            A2 = vld1q_f32(pnt_a2);
            C2 = vfmaq_laneq_f32(C2,B,A2,0);
            A3 = vld1q_f32(pnt_a3);
            C3 = vfmaq_laneq_f32(C3,B,A3,0);

            B = vld1q_f32(&B(p+1,0));
            C0 = vfmaq_laneq_f32(C0,B,A0,1);
            C1 = vfmaq_laneq_f32(C1,B,A1,1);
            C2 = vfmaq_laneq_f32(C2,B,A2,1);
            C3 = vfmaq_laneq_f32(C3,B,A3,1);

            B = vld1q_f32(&B(p+2,0));
            C0 = vfmaq_laneq_f32(C0,B,A0,2);
            C1 = vfmaq_laneq_f32(C1,B,A1,2);
            C2 = vfmaq_laneq_f32(C2,B,A2,2);
            C3 = vfmaq_laneq_f32(C3,B,A3,2);

            B = vld1q_f32(&B(p+3,0));
            C0 = vfmaq_laneq_f32(C0,B,A0,3);
            C1 = vfmaq_laneq_f32(C1,B,A1,3);
            C2 = vfmaq_laneq_f32(C2,B,A2,3);
            C3 = vfmaq_laneq_f32(C3,B,A3,3);
            
            pnt_a0 +=4;
            pnt_a1 +=4;
            pnt_a2 +=4;
            pnt_a3 +=4;
        }
        vst1q_f32(&C(0,0),C0);
        vst1q_f32(&C(1,0),C1);
        vst1q_f32(&C(2,0),C2);
        vst1q_f32(&C(3,0),C3);
    }
}
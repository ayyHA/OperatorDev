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
    void add_dot4x4_packed(int k,float* a,int lda,float* b,int ldb,float* c,int ldc){
        // float* pnt_a_0p = a, *pnt_a_1p = a+1, *pnt_a_2p = a+2, *pnt_a_3p = a+3;
        float* c00 = &C(0,0);
        float32x4_t C0 = vld1q_f32(c00),C1 = vld1q_f32(c00+ldc), C2 = vld1q_f32(c00+2*ldc), C3 = vld1q_f32(c00+3*ldc);
        // 切成下面这种被注释的写法,咋会有误差,而且将fma换成mla会很慢
        // float32x4_t C0  = {0},C1 = {0},C2 = {0},C3 = {0};
        float32x4_t B;
        int p;
        for(p=0;p<k;p++){
            B = vld1q_f32(b);
            b+=4;
            C0 = vfmaq_n_f32(C0,B,*a);
            C1 = vfmaq_n_f32(C1,B,*(a+1));
            C2 = vfmaq_n_f32(C2,B,*(a+2));
            C3 = vfmaq_n_f32(C3,B,*(a+3));
            a+=4;
        }
        // float32x4_t C00 = vld1q_f32(c00);
        // C00 = vaddq_f32(C00,C0);
        // vst1q_f32(c00,C00);

        // c00 = &C(1,0);
        // C00 = vld1q_f32(c00);
        // C00 = vaddq_f32(C00,C1);
        // vst1q_f32(c00,C00);

        // c00 = &C(2,0);
        // C00 = vld1q_f32(c00);
        // C00 = vaddq_f32(C00,C2);
        // vst1q_f32(c00,C00);

        // c00 = &C(3,0);
        // C00 = vld1q_f32(c00);
        // C00 = vaddq_f32(C00,C3);
        // vst1q_f32(c00,C00);

        vst1q_f32(c00,C0);
        vst1q_f32(c00+ldc,C1);
        vst1q_f32(c00+2*ldc,C2);
        vst1q_f32(c00+3*ldc,C3);        
    }

    void packedA_block(int k,float* a,int lda,float* packedA){
        float* pnt_a00 = a,*pnt_a10 = a+lda,*pnt_a20 = a+lda*2,*pnt_a30 = a+lda*3;
        int p;
        for(p=0;p<k;p++){
            *packedA++ = *pnt_a00++;
            *packedA++ = *pnt_a10++;
            *packedA++ = *pnt_a20++;
            *packedA++ = *pnt_a30++;
        }
    }

    void packedB_panel(int k,float* b,int ldb,float* packedB){
        float* pnt_bp0;
        int p;
        for(p=0;p<k;p++){
            pnt_bp0 = b+p*ldb;
            *packedB++ = *pnt_bp0++;
            *packedB++ = *pnt_bp0++;
            *packedB++ = *pnt_bp0++;
            *packedB++ = *pnt_bp0++; 
        }
    }

    // void add_dot4x4(int,float*,int,float*,int,float*,int);
    void gemm_4x4_neon_packedAB(int m,int n,int k,float* a,int lda,
                        float* b,int ldb,float* c,int ldc){
        // 用于A子块打包,以使得内存连续,排布方式符合我们的操作
        float* packedA = new float[m*k];
        // 用于B子块打包
        float* packedB = new float[k*n];
        
        int i,j;
        for(j=0;j<n;j+=4){
            packedB_panel(k,&B(0,j),ldb,packedB+j*k);
            for(i=0;i<m;i+=4){
                if(j==0){
                    packedA_block(k,&A(i,0),lda,packedA+i*k);
                }
                add_dot4x4_packed(k,packedA+i*k,4,packedB+j*k,4,&C(i,j),ldc);
            }
        }
    }
}

// M维的分块
#define mc 256
// K维的分块
#define kc 128
/*
    上面是针对一个矩阵划分成4x4的小块,利用neon intrinsic进行计算(bbuf说是按照"z轴"分块)
    而对于一个很大的A,B阵,实际应该对它们先分块,然后再调用上面的函数进行4x4小块的计算(bbuf说是按照"行列"再进行分块)
*/
void gemm_4x4_neon_packed(int m,int n,int k,float* a,int lda,float* b,int ldb,float*c,int ldc){
    int i,p,pb,ib;
    for(p=0;p<k;p+=kc){
        // 分块后k维的大小,考虑边界因此有min
        pb = std::min(kc,k-p);
        for(i=0;i<m;i+=mc){
            // 分块后的m维的大小,考虑边界因此有min
            ib = std::min(mc,m-i);
            neon_4x4::gemm_4x4_neon_packedAB(ib,n,pb,&A(i,p),lda,&B(p,0),ldb,&C(i,0),ldc);
        }
    }
}
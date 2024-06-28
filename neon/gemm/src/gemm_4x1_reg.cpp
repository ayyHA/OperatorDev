#include "gemm.h"

#define A(i,j) a[(i)*(lda) + (j)]
#define B(i,j) b[(i)*(ldb) + (j)]
#define C(i,j) c[(i)*(ldc) + (j)]

namespace reg_4x1{
    void add_dot4x1(int,float*,int,float*,int,float*,int);
    void add_dot4x1_unroll(int,float*,int,float*,int,float*,int);
}

void gemm_4x1_reg(int m,int n,int k,float* a,int lda,
                float* b,int ldb,float* c,int ldc){
    int i,j;
    for(i=0;i<m;i+=4){
        for(j=0;j<n;j++){
            // reg_4x1::add_dot4x1(k,&A(i,0),lda,&B(0,j),ldb,&C(i,j),ldc);
            reg_4x1::add_dot4x1_unroll(k,&A(i,0),lda,&B(0,j),ldb,&C(i,j),ldc);
        }
    }
}

namespace reg_4x1{
    void add_dot4x1(int k,float* a,int lda,float* b,int ldb,float* c,int ldc){
        register float reg_b_p0;
        register float reg_c_00 = C(0,0), reg_c_10 = C(1,0), reg_c_20 = C(2,0), reg_c_30 = C(3,0);
        float* pnt_a_0p = &A(0,0),*pnt_a_1p = &A(1,0),*pnt_a_2p = &A(2,0),*pnt_a_3p = &A(3,0);

        int p;
        for(p=0;p<k;p++){
            reg_b_p0 = B(p,0);
            reg_c_00 += *pnt_a_0p++ * reg_b_p0; 
            reg_c_10 += *pnt_a_1p++ * reg_b_p0;
            reg_c_20 += *pnt_a_2p++ * reg_b_p0;
            reg_c_30 += *pnt_a_3p++ * reg_b_p0;
        }
        C(0,0) = reg_c_00;
        C(1,0) = reg_c_10;
        C(2,0) = reg_c_20;
        C(3,0) = reg_c_30;
    }

    // unroll=4
    void add_dot4x1_unroll(int k,float* a,int lda,float* b,int ldb,float* c,int ldc){
        register float reg_b_p0;
        register float reg_c_00 = C(0,0),reg_c_10 = C(1,0),reg_c_20 = C(2,0),reg_c_30 = C(3,0);
        float* pnt_a_0p = &A(0,0), *pnt_a_1p = &A(1,0), *pnt_a_2p = &A(2,0), *pnt_a_3p = &A(3,0);

        int p;
        for(p=0;p<k;p+=4){
            reg_b_p0 = B(p,0);
            reg_c_00 += *pnt_a_0p * reg_b_p0;
            reg_c_10 += *pnt_a_1p * reg_b_p0;
            reg_c_20 += *pnt_a_2p * reg_b_p0;
            reg_c_30 += *pnt_a_3p * reg_b_p0;

            reg_b_p0 = B(p+1,0);
            reg_c_00 += *(pnt_a_0p+1) * reg_b_p0;
            reg_c_10 += *(pnt_a_1p+1) * reg_b_p0;
            reg_c_20 += *(pnt_a_2p+1) * reg_b_p0;
            reg_c_30 += *(pnt_a_3p+1) * reg_b_p0;

            reg_b_p0 = B(p+2,0);
            reg_c_00 += *(pnt_a_0p+2) * reg_b_p0;
            reg_c_10 += *(pnt_a_1p+2) * reg_b_p0;
            reg_c_20 += *(pnt_a_2p+2) * reg_b_p0;
            reg_c_30 += *(pnt_a_3p+2) * reg_b_p0;

            reg_b_p0 = B(p+3,0);
            reg_c_00 += *(pnt_a_0p+3) * reg_b_p0;
            reg_c_10 += *(pnt_a_1p+3) * reg_b_p0;
            reg_c_20 += *(pnt_a_2p+3) * reg_b_p0;
            reg_c_30 += *(pnt_a_3p+3) * reg_b_p0;

            pnt_a_0p+=4;
            pnt_a_1p+=4;
            pnt_a_2p+=4;
            pnt_a_3p+=4;
        }
        C(0,0) = reg_c_00;
        C(1,0) = reg_c_10;
        C(2,0) = reg_c_20;
        C(3,0) = reg_c_30;
    }
}   // reg_4x1 namespace 
#include "gemm.h"

#define A(i,j) a[(i)*(lda) + (j)]
#define B(i,j) b[(i)*(ldb) + (j)]
#define C(i,j) c[(i)*(ldc) + (j)]

namespace reg_4x4{
    void add_dot4x4(int,float*,int,float*,int,float*,int);
}

void gemm_4x4_reg(int m,int n,int k,float* a,int lda,
                float* b,int ldb,float* c,int ldc){
    int i,j;
    for(i=0;i<m;i+=4){
        for(j=0;j<n;j+=4){
            reg_4x4::add_dot4x4(k,&A(i,0),lda,&B(0,j),ldb,&C(i,j),ldc);
        }
    }
}

namespace reg_4x4{
    void add_dot4x4(int k,float* a,int lda,float* b,int ldb,float* c,int ldc){
        register float reg_c_00 = C(0,0),reg_c_01 = C(0,1),reg_c_02 = C(0,2),reg_c_03 = C(0,3),
                    reg_c_10 = C(1,0),reg_c_11 = C(1,1),reg_c_12 = C(1,2),reg_c_13 = C(1,3),
                    reg_c_20 = C(2,0),reg_c_21 = C(2,1),reg_c_22 = C(2,2),reg_c_23 = C(2,3),
                    reg_c_30 = C(3,0),reg_c_31 = C(3,1),reg_c_32 = C(3,2),reg_c_33 = C(3,3);

        register float reg_b_p0,reg_b_p1,reg_b_p2,reg_b_p3;
        register float reg_a_0p,reg_a_1p,reg_a_2p,reg_a_3p;

        float* pnt_a_0p = &A(0,0),*pnt_a_1p = &A(1,0),*pnt_a_2p = &A(2,0),*pnt_a_3p = &A(3,0);
        int p;

        for(p=0;p<k;p++){
            reg_a_0p = *pnt_a_0p++;
            reg_a_1p = *pnt_a_1p++;
            reg_a_2p = *pnt_a_2p++;
            reg_a_3p = *pnt_a_3p++;

            reg_b_p0 = B(p,0);
            reg_c_00 += reg_a_0p * reg_b_p0;
            reg_c_10 += reg_a_1p * reg_b_p0;
            reg_c_20 += reg_a_2p * reg_b_p0;
            reg_c_30 += reg_a_3p * reg_b_p0;

            reg_b_p1 = B(p,1);
            reg_c_01 += reg_a_0p * reg_b_p1;
            reg_c_11 += reg_a_1p * reg_b_p1;
            reg_c_21 += reg_a_2p * reg_b_p1;
            reg_c_31 += reg_a_3p * reg_b_p1;

            reg_b_p2 = B(p,2);
            reg_c_02 += reg_a_0p * reg_b_p2;
            reg_c_12 += reg_a_1p * reg_b_p2;
            reg_c_22 += reg_a_2p * reg_b_p2;
            reg_c_32 += reg_a_3p * reg_b_p2;

            reg_b_p3 = B(p,3);
            reg_c_03 += reg_a_0p * reg_b_p3;
            reg_c_13 += reg_a_1p * reg_b_p3;
            reg_c_23 += reg_a_2p * reg_b_p3;
            reg_c_33 += reg_a_3p * reg_b_p3;      
        }
        C(0,0) = reg_c_00,  C(0,1) = reg_c_01,  C(0,2) = reg_c_02,  C(0,3) = reg_c_03;
        C(1,0) = reg_c_10,  C(1,1) = reg_c_11,  C(1,2) = reg_c_12,  C(1,3) = reg_c_13;
        C(2,0) = reg_c_20,  C(2,1) = reg_c_21,  C(2,2) = reg_c_22,  C(2,3) = reg_c_23;
        C(3,0) = reg_c_30,  C(3,1) = reg_c_31,  C(3,2) = reg_c_32,  C(3,3) = reg_c_33;
    }
}
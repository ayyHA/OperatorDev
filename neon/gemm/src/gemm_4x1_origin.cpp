#include "gemm.h"

#define A(i,j) a[(i)*lda + (j)]
#define B(i,j) b[(i)*ldb + (j)]
#define C(i,j) c[(i)*ldc + (j)]

namespace origin_4x1{
    void add_dot4x1(int,float*,int,float*,int,float*,int);
}

void gemm_4x1_origin(int m,int n,int k,float* a,int lda,
                    float* b,int ldb,float* c,int ldc){
    int i,j;
    for(i=0;i<m;i+=4){
        for(j=0;j<n;j++){
            origin_4x1::add_dot4x1(k,&A(i,0),lda,&B(0,j),ldb,&C(i,j),ldc);
        }
    }
}

namespace origin_4x1{
    void add_dot4x1(int k,float* a,int lda,float* b,int ldb,float*c,int ldc){
        int p;
        for(p=0;p<k;p++){
            C(0,0) += A(0,p) * B(p,0);
            C(1,0) += A(1,p) * B(p,0);
            C(2,0) += A(2,p) * B(p,0);
            C(3,0) += A(3,p) * B(p,0);
        }
    }
}
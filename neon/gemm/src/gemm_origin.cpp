#include "gemm.h"

#define A(i,j) a[(i)*(lda)+(j)]
#define B(i,j) b[(i)*(ldb)+(j)]
#define C(i,j) c[(i)*(ldc)+(j)]

void gemm_origin(int m,int n,int k,float* a,int lda,float* b,int ldb,float*c,int ldc){
    int i,j,p;
    float sum=0;
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            sum=0;
            for(p=0;p<k;p++){
                // C(i,j) = C(i,j) + A(i,p) * B(p,j);
                sum += A(i,p) * B(p,j);
            }
            C(i,j) += sum;
        }
    }
}
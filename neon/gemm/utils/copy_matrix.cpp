#define A(i,j) a[(i)*(lda) + (j)]
#define B(i,j) b[(i)*(ldb) + (j)]

void copy_matrix(int m,int n,float*a,int lda,float*b,int ldb){
    for(int i=0;i<m;i++)
        for(int j=0;j<n;j++)
            B(i,j) = A(i,j);
}
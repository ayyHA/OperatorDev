#define A(i,j) a[(i)*(lda) + (j)]
#define B(i,j) b[(i)*(ldb) + (j)]
#define abs(i) (i) > 0 ? (i) : -(i)

double compare_matrix(int m,int n,float* a,int lda,float* b,int ldb){
    double max_diff=.0, diff;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            diff = abs(A(i,j) - B(i,j));
            max_diff = max_diff>diff?max_diff:diff;
        }
    }
    return max_diff;
}
#include <cstdlib>
#include <ctime>
#define A(i,j) a[(i)*(lda) + (j)]

void random_matrix(int m, int n, float* a, int lda){
    srand48(time(NULL));
    for(int i=0;i<m;i++)
        for(int j=0;j<n;j++)
            A(i,j) = drand48();
}
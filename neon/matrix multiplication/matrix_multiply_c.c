/*
    这里考虑的是列主序的矩阵乘
    C=A*B
    C的size是n*m
    C(i,j) = C[i+j*ldc] = C[i+j*n] 
*/
// #define C(i,j) C[i+j*n]
void matrix_multiply_c(float32_t* A,float32_t* B,float32_t* C,uint32_t n,uint32_t m,uint32_t k){
    int i,j,p;
    for(i=0;i<n;i++){
        for(j=0;j<m;j++){
            C[i+j*n] = 0;
            // Inner Product
            for(p=0;p<k;p++){
                C[i+j*n] += A[i + p*n] * B[p + j*k]; 
            }
        }
    }
}


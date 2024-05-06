#define A(i, j) a[(i) + (j)*lda]
#define B(i, j) b[(i) + (j)*ldb]
#define C(i, j) c[(i) + (j)*ldc]

void col_sgemm(int m, int n, int k,
               double *a, int lda,
               double *b, int ldb,
               double *c, int ldc)
{
    int i, j;
    for (j = 0; j < n; j += 4)
    {
        for (i = 0; i < m; i++)
            add_dot1x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
}

void add_dot1x4(int k,
                double *a, int lda,
                double *b, int ldb,
                double *c, int ldc)
{
    // 3. 将原先的add_dot这个函数内联进来,减少不必要的函数调用开销;
    // 并且将它们写在一个for里,一次做4个内积,减少循环次数,并且利用了空间局部性
    int p;
    for (p = 0; p < k; p++)
    {
        C(0, 0) += A(0, p) * B(p, 0);
        C(0, 1) += A(0, p) * B(p, 1);
        C(0, 2) += A(0, p) * B(p, 2);
        C(0, 3) += A(0, p) * B(p, 3);
    }
}
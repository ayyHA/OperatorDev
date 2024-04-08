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
        {
            add_dot1x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

void add_dot1x4(int k,
                double *a, int lda,
                double *b, int ldb,
                double *c, int ldc)
{
    // 4.用寄存器存储要进行频繁运算的数据
    register double reg_c00 = .0, reg_c01 = .0, reg_c02 = .0, reg_c03 = .0, reg_a0p;
    // 5. B采用指针,减少宏替换(减少浮点操作次数应该是)
    double *pnt_bp0, *pnt_bp1, *pnt_bp2, *pnt_bp3;
    pnt_bp0 = &B(0, 0);
    pnt_bp1 = &B(0, 1);
    pnt_bp2 = &B(0, 2);
    pnt_bp3 = &B(0, 3);

    int p;
    for (p = 0; p < k; p++)
    {
        reg_a0p = A(0, p);
        reg_c00 += reg_a0p * *pnt_bp0++;
        reg_c01 += reg_a0p * *pnt_bp1++;
        reg_c02 += reg_a0p * *pnt_bp2++;
        reg_c03 += reg_a0p * *pnt_bp3++;
    }
    C(0, 0) += reg_c00;
    C(0, 1) += reg_c01;
    C(0, 2) += reg_c02;
    C(0, 3) += reg_c03;
}
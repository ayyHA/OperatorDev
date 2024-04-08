#define A(i, j) a[(i) + (j)*lda]
#define B(i, j) b[(i) + (j)*ldb]
#define C(i, j) c[(i) + (j)*ldc]

void col_sgemm(int m, int n, int k,
               double *a, int lda,
               double *b, int ldb,
               double *c, int ldc)
{
    int i, j;
    for (int j = 0; j < n; j += 4)
    {
        for (int i = 0; i < m; i++)
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
    int p;
    register double reg_c00 = .0, reg_c01 = .0, reg_c02 = .0, reg_c03 = .0, reg_a0p;
    double *ptn_bp0, *ptn_bp1, *ptn_bp2, *ptn_bp3;
    ptn_bp0 = &B(0, 0);
    ptn_bp1 = &B(0, 1);
    ptn_bp2 = &B(0, 2);
    ptn_bp3 = &B(0, 3);

    // 6. 循环展开
    // 7. 循环展开中B的数据采用间接取址(这个不知道啥鸟蛋作用)
    for (p = 0; p < k; p += 4)
    { // p incr 是4,展开4个
        reg_a0p = A(0, p);
        reg_c00 += reg_a0p * *ptn_bp0;
        reg_c01 += reg_a0p * *ptn_bp1;
        reg_c02 += reg_a0p * *ptn_bp2;
        reg_c03 += reg_a0p * *ptn_bp3;

        reg_a0p = A(0, p + 1);
        reg_c00 += reg_a0p * *(ptn_bp0 + 1);
        reg_c01 += reg_a0p * *(ptn_bp1 + 1);
        reg_c02 += reg_a0p * *(ptn_bp2 + 1);
        reg_c03 += reg_a0p * *(ptn_bp3 + 1);

        reg_a0p = A(0, p + 2);
        reg_c00 += reg_a0p * *(ptn_bp0 + 2);
        reg_c01 += reg_a0p * *(ptn_bp1 + 2);
        reg_c02 += reg_a0p * *(ptn_bp2 + 2);
        reg_c03 += reg_a0p * *(ptn_bp3 + 2);

        reg_a0p = A(0, p + 3);
        reg_c00 += reg_a0p * *(ptn_bp0 + 3);
        reg_c01 += reg_a0p * *(ptn_bp1 + 3);
        reg_c02 += reg_a0p * *(ptn_bp2 + 3);
        reg_c03 += reg_a0p * *(ptn_bp3 + 3);

        ptn_bp0 += 4;
        ptn_bp1 += 4;
        ptn_bp2 += 4;
        ptn_bp3 += 4;
    }
    C(0, 0) += reg_c00;
    C(0, 1) += reg_c01;
    C(0, 2) += reg_c02;
    C(0, 3) += reg_c03;
}
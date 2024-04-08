#define A(i, j) a[(i) + (j)*lda]
#define B(i, j) b[(i) + (j)*ldb]
#define C(i, j) c[(i) + (j)*ldc]

/**
 * 1. 这里展示的是一次进行4x4这么一个子阵的矩阵乘,意味着一次会做16次内积;
 * 与gemm3_1x4类似,内联了add_dot(单次内积的调用函数),并且合并了16次内积于一个循环里
 * 同时采用寄存器将频繁使用的值存入,在gemm4_1x4中,B子阵的相关参数不被存入,因为每个只用了一次就划到下一行,
 * 但在4x4子阵中因需频繁使用,因此也放入寄存器中
 */
/**
 *需要注意这里没考虑矩阵是否能被4整除,即是否需要padding的问题
 */
void col_sgemm(int m, int n, int k,
               double *a, int lda,
               double *b, int ldb,
               double *c, int ldc)
{
    int i, j;
    for (j = 0; j < n; j += 4)
    {
        for (i = 0; i < m; i += 4)
        {
            add_dot4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

void add_dot4x4(int k,
                double *a, int lda,
                double *b, int ldb,
                double *c, int ldc)
{
    int p;
    register double reg_c00 = .0, reg_c01 = .0, reg_c02 = .0, reg_c03 = .0,
                    reg_c10 = .0, reg_c11 = .0, reg_c12 = .0, reg_c13 = .0,
                    reg_c20 = .0, reg_c21 = .0, reg_c22 = .0, reg_c23 = .0,
                    reg_c30 = .0, reg_c31 = .0, reg_c32 = .0, reg_c33 = .0;
    register double reg_a0p, reg_a1p, reg_a2p, reg_a3p;
    register double reg_bp0, reg_bp1, reg_bp2, reg_bp3;
    double *pnt_bp0, *pnt_bp1, *pnt_bp2, *pnt_bp3;

    pnt_bp0 = &B(0, 0);
    pnt_bp1 = &B(0, 1);
    pnt_bp2 = &B(0, 2);
    pnt_bp3 = &B(0, 3);

    for (p = 0; p < k; p++)
    {
        reg_bp0 = *pnt_bp0++;
        reg_bp1 = *pnt_bp1++;
        reg_bp2 = *pnt_bp2++;
        reg_bp3 = *pnt_bp3++;

        reg_a0p = A(0, p);
        reg_c00 += reg_a0p * reg_bp0;
        reg_c01 += reg_a0p * reg_bp1;
        reg_c02 += reg_a0p * reg_bp2;
        reg_c03 += reg_a0p * reg_bp3;

        reg_a1p = A(1, p);
        reg_c10 += reg_a1p * reg_bp0;
        reg_c11 += reg_a1p * reg_bp1;
        reg_c12 += reg_a1p * reg_bp2;
        reg_c13 += reg_a1p * reg_bp3;

        reg_a2p = A(2, p);
        reg_c20 += reg_a2p * reg_bp0;
        reg_c21 += reg_a2p * reg_bp1;
        reg_c22 += reg_a2p * reg_bp2;
        reg_c23 += reg_a2p * reg_bp3;

        reg_a3p = A(3, p);
        reg_c30 += reg_a3p * reg_bp0;
        reg_c31 += reg_a3p * reg_bp1;
        reg_c32 += reg_a3p * reg_bp2;
        reg_c33 += reg_a3p * reg_bp3;
    }
    C(0, 0) += reg_c00, C(0, 1) += reg_c01, C(0, 2) += reg_c02, C(0, 3) += reg_c03;
    C(1, 0) += reg_c10, C(1, 1) += reg_c11, C(1, 2) += reg_c12, C(1, 3) += reg_c13;
    C(2, 0) += reg_c20, C(2, 1) += reg_c21, C(2, 2) += reg_c22, C(2, 3) += reg_c23;
    C(3, 0) += reg_c30, C(3, 1) += reg_c31, C(3, 2) += reg_c32, C(3, 3) += reg_c33;
}
#define A(i, j) a[i + lda * j]
#define B(i, j) b[i + ldb * j]
#define C(i, j) c[i + ldc * j]

// 1.这里展示的是列主序的串行的通用矩阵乘的内积乘法,是最朴素的一种方法;
//   将内积的乘法单独提取到add_dot这个function里
void col_sgemm(int m, int n, int k,
               double *a, int lda,
               double *b, int ldb,
               double *c, int ldc)
{
    int i, j, p;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            /* 下面的这种是最通常的写法 */
            // for (p = 0; p < k; p++)
            // {
            //     C(i, j) += A(i, p) * B(p, j);
            // }

            /* 换成对应的内积函数处理 */
            add_dot(k, &A(i, 0), lda, &B(0, j), &C(i, j));
        }
    }
}

#define X(p) x[p * lda]
void add_dot(int k, double *x, int lda, double *y, double *ans)
{
    int p;
    for (p = 0; p < k; p++)
    {
        *ans += X(p) * y[p];
    }
}
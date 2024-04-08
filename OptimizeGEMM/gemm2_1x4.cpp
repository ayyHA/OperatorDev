#define A(i, j) a[(i) + (j)*lda]
#define B(i, j) b[(i) + (j)*ldb]
#define C(i, j) c[(i) + (j)*ldc]

// 2. 将内积由一次做一个变成一次做四个,通过循环展开的方式实现,交由add_dot1x4函数实现
void col_sgemm(int m, int n, int k,
               double *a, int lda,
               double *b, int ldb,
               double *c, int ldc)
{
    int i, j;
    // 调换了一下遍历顺序:n -> m,以减少一点cache miss,调整j的步距,使得一次调用做一次内积,变成做四次
    // j:[j,j+1,j+2,j+3]  size(j): 4
    for (j = 0; j < n; j += 4)
    {
        for (i = 0; i < m; i++)
        {
            add_dot1x4(m, n, k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

void add_dot1x4(int m, int n, int k,
                double *a, int lda,
                double *b, int ldb,
                double *c, int ldc)
{
    // 所谓做4次内积就是调用4次add_dot
    // 这里输入的是地址,所以相当于以那个输入的地址为起点做内积(相当于子矩阵),因此都是0,1,2,3这样的常量
    /* 为什么不用循环,而是直接一坨 */
    add_dot(k, &A(0, 0), lda, &B(0, 0), &C(0, 0));
    add_dot(k, &A(0, 0), lda, &B(0, 1), &C(0, 1));
    add_dot(k, &A(0, 0), lda, &B(0, 2), &C(0, 2));
    add_dot(k, &A(0, 0), lda, &B(0, 3), &C(0, 3));
}

#define X(i) x[i * lda]
void add_dot(int k, double *x, int lda, double *y, double *ans)
{
    int p;
    for (p = 0; p < k; p++)
        *ans += X(p) * y[p];
}
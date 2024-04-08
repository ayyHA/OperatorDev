#include "intrin.h"

#define A(i, j) a[(i) + (j)*lda]
#define B(i, j) b[(i) + (j)*ldb]
#define C(i, j) c[(i) + (j)*ldc]

#define min(i, j) ((i) < (j) ? (i) : (j))
/**
 * 3.给B的分块也进行打包，同时避免重复的进行数据打包
 */
const int mc = 256, kc = 128;
static int b_i = 0;
// 列主序，内部用内积
void col_sgemm(int m, int n, int k,
               double *a, int lda,
               double *b, int ldb,
               double *c, int ldc)
{
    int i, p;
    int ib, pb;
    for (p = 0; p < k; p += kc)
    {
        pb = min(k - p, kc);
        double *packed_b;
        for (i = 0; i < m; i += mc)
        {
            ib = min(m - i, mc);
            if (i == 0)
                packed_b = new double[k * n];
            inner_kernel(ib, n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc, i == 0, packed_b);
        }
        delete packed_b;
    }
}

// 划分后的子块
void inner_kernel(int m, int n, int k,
                  double *a, int lda,
                  double *b, int ldb,
                  double *c, int ldc, bool flag, double *packed_b)
{
    double *packed_a = new double[m * k];
    // double *packed_b = new double[k * n]; // 这个地址其实需要放在上一层循环,最好用智能指针,不然处理起来太麻烦了,很难避免内存泄露
    int i, j;
    for (j = 0; j < n; j += 4)
    {
        if (flag) // flag为true，说明是上一层循环中i==0的时刻，刚进入一个新的位置&B(p,0)，需要读入该k*n个的新数据
            pack_matrix_b(k, &B(0, j), ldb, &packed_b[j * k]);
        for (i = 0; i < m; i += 4)
        {
            if (j == 0) // 此时会将A子阵扫一遍，放入packed_a中
                pack_matrix_a(k, &A(i, 0), lda, &packed_a[i * k]);
            // 为C阵的4*4区域做内积，而此刻需要做内积的部分，我们已经给它打包进各自的数组
            // 给定的打包数组标明的是起始地址，各自均是4*128=512个元素
            add_dot4x4(k, &packed_a[i * k], 4, &packed_b[j * k], 4, &C(i, j), ldc);
        }
    }
    delete packed_a;
}

// 给A阵的子块打包
void pack_matrix_a(int k, double *a, int lda, double *packed_a)
{
    int p;
    double *pnt_a;
    for (p = 0; p < k; p++) // k==128
    {
        pnt_a = &A(0, p);       // 显式将pnt_a的指针移到正确的位置
        *packed_a++ = *pnt_a++; // 执行四次赋值操作，并移动指针，其中最后一次的pnt_a的位置是错的，需要上一行代码显式挪动
        *packed_a++ = *pnt_a++;
        *packed_a++ = *pnt_a++;
        *packed_a++ = *pnt_a++;
    }
}

// 给B阵的子块打包
void pack_matrix_b(int k, double *b, int ldb, double *packed_b)
{
    int p;
    double *pnt_b0 = &B(0, 0), *pnt_b1 = &B(0, 1),
           *pnt_b2 = &B(0, 2), *pnt_b3 = &B(0, 3);
    for (p = 0; p < k; p++) // k==128
    {
        *packed_b++ = *pnt_b0++;
        *packed_b++ = *pnt_b1++;
        *packed_b++ = *pnt_b2++;
        *packed_b++ = *pnt_b3++;
    }
}

// 对两个打包的进行运算，生成4x4的结果子阵，即做了16次内积，内部通过intrinsic function进行实现
void add_dot4x4(int k,
                double *a, int lda,
                double *b, int ldb,
                double *c, int ldc)
{
    __m128d vc_00_10, vc_01_11, vc_02_12, vc_03_13,
        vc_20_30, vc_21_31, vc_22_32, vc_23_33;

    __m128d va_0p_1p, va_2p_3p;
    __m128d vb_p0, vb_p1, vb_p2, vb_p3;

    // 给C阵的第0,1行初始化
    vc_00_10 = _mm_set1_pd(0);
    vc_01_11 = _mm_set1_pd(0);
    vc_02_12 = _mm_set1_pd(0);
    vc_03_13 = _mm_set1_pd(0);
    // 给C阵的第2,3行初始化
    vc_20_30 = _mm_set1_pd(0);
    vc_21_31 = _mm_set1_pd(0);
    vc_22_32 = _mm_set1_pd(0);
    vc_23_33 = _mm_set1_pd(0);

    int p;
    for (p = 0; p < k; p++)
    {
        va_0p_1p = _mm_load_pd(a);
        va_2p_3p = _mm_load_pd(a + 2);
        a += 4;

        vb_p0 = _mm_loaddup_pd(b);
        vb_p1 = _mm_loaddup_pd(b + 1);
        vb_p2 = _mm_loaddup_pd(b + 2);
        vb_p3 = _mm_loaddup_pd(b + 3);
        b += 4;
        // 因为没有乘加,所以自己乘加
        vc_00_10 = _mm_add_pd(vc_00_10, _mm_mul_pd(va_0p_1p, vb_p0));
        vc_01_11 = _mm_add_pd(vc_01_11, _mm_mul_pd(va_0p_1p, vb_p1));
        vc_02_12 = _mm_add_pd(vc_02_12, _mm_mul_pd(va_0p_1p, vb_p2));
        vc_03_13 = _mm_add_pd(vc_03_13, _mm_mul_pd(va_0p_1p, vb_p3));

        vc_20_30 = _mm_add_pd(vc_20_30, _mm_mul_pd(va_2p_3p, vb_p0));
        vc_21_31 = _mm_add_pd(vc_21_31, _mm_mul_pd(va_2p_3p, vb_p1));
        vc_22_32 = _mm_add_pd(vc_22_32, _mm_mul_pd(va_2p_3p, vb_p2));
        vc_23_33 = _mm_add_pd(vc_23_33, _mm_mul_pd(va_2p_3p, vb_p3));
    }
    _mm_store_pd(&C(0, 0), vc_00_10);
    _mm_store_pd(&C(0, 1), vc_01_11);
    _mm_store_pd(&C(0, 2), vc_02_12);
    _mm_store_pd(&C(0, 3), vc_03_13);

    _mm_store_pd(&C(2, 0), vc_20_30);
    _mm_store_pd(&C(2, 1), vc_21_31);
    _mm_store_pd(&C(2, 2), vc_22_32);
    _mm_store_pd(&C(2, 3), vc_23_33);
}
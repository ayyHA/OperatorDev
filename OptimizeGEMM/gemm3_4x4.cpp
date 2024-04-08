#include "intrin.h"

#define A(i, j) a[(i) + (j)*lda]
#define B(i, j) b[(i) + (j)*ldb]
#define C(i, j) c[(i) + (j)*ldc]

#define min(i, j) ((i) < (j) ? (i) : (j))

/**
 * 2. 增加分块和打包内容
 * 分块,子块大小： kc:128,mc:256
 * 也就是A的子块：256*128，B的子块：128*1024
 *
 * 打包,估计就是把一些子块的子阵，比如A的子块的子阵：4*128弄成一个连续的内存块
 */
const int mc = 256, kc = 128;
void col_sgemm(int m, int n, int k,
               double *a, int lda,
               double *b, int ldb,
               double *c, int ldc)
{
    int i, p;
    int ib, pb; // i_block,p_block 这个是防止最后的块不足mc,kc这些大小
    for (p = 0; p < k; p += kc)
    {
        pb = min(k - p, kc);
        for (i = 0; i < m; i += mc)
        {
            ib = min(m - i, mc);
            // 划分子块,子块的size用ib,n,pb而非mc,n,kc
            inner_kernel(ib, n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc);
        }
    }
}

/**
 * 子块需要打包和计算
 * 计算是16个内积，采用之前的add_dot4x4
 *
 */
void inner_kernel(int m, int n, int k,
                  double *a, int lda,
                  double *b, int ldb,
                  double *c, int ldc)
{
    double block_a[mc * kc];
    int i, j;
    for (j = 0; j < n; j += 4)
    {
        for (i = 0; i < m; i += 4)
        {
            if (j == 0) // 填充完毕一次即可，后面重复利用
                pack_matrix_a(k, &A(i, 0), lda, &block_a[i * kc]);
            add_dot4x4(k, &block_a[i * kc], 4, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

void pack_matrix_a(int k, double *a, int lda, double *block_a)
{
    int p;
    double *pnt_a;
    for (p = 0; p < k; p++)
    {
        pnt_a = &A(0, p);
        *block_a++ = *pnt_a++;
        *block_a++ = *pnt_a++;
        *block_a++ = *pnt_a++;
        *block_a++ = *pnt_a++;
    }
}

/**
 * 采用联合体节省内存，方便后面将值store回内存，但其实没用到vec_2_d(vector to double)这个属性
 */
union vec_2_d
{
    __m128d vec;
    double d[2];
    vec_2_d() : vec(), d(){};
};

/**
 * 16个内积
 * _mm_loaddup_pd(double*) 将double*指向的内存的值广播到整个向量寄存器中
 * 这里采用的是128位的xmm寄存器，用的是SSE指令集的向量函数
 */
void add_dot4x4(int k,
                double *a, int lda,
                double *b, int ldb,
                double *c, int ldc)
{
    int p;
    vec_2_d vec_c_00_10, vec_c_01_11, vec_c_02_12, vec_c_03_13,
        vec_c_20_30, vec_c_21_31, vec_c_22_32, vec_c_23_33;
    vec_2_d vec_a_0p_1p, vec_a_2p_3p;
    vec_2_d vec_b_p0, vec_b_p1, vec_b_p2, vec_b_p3;

    vec_c_00_10.vec = _mm_setzero_pd();
    vec_c_01_11.vec = _mm_setzero_pd();
    vec_c_02_12.vec = _mm_setzero_pd();
    vec_c_03_13.vec = _mm_setzero_pd();
    vec_c_20_30.vec = _mm_setzero_pd();
    vec_c_21_31.vec = _mm_setzero_pd();
    vec_c_22_32.vec = _mm_setzero_pd();
    vec_c_23_33.vec = _mm_setzero_pd();

    double *pnt_bp0, *pnt_bp1, *pnt_bp2, *pnt_bp3;
    pnt_bp0 = &B(0, 0);
    pnt_bp1 = &B(0, 1);
    pnt_bp2 = &B(0, 2);
    pnt_bp3 = &B(0, 3);

    for (p = 0; p < k; p++)
    {
        // a的初始化，采用打包好的数组来取代&A(0,p),&A(2,p)
        vec_a_0p_1p.vec = _mm_load_pd(a);
        vec_a_2p_3p.vec = _mm_load_pd(a + 2);
        // A打包后的步距
        a += 4;

        // b的初始化
        vec_b_p0.vec = _mm_loaddup_pd(pnt_bp0++);
        vec_b_p1.vec = _mm_loaddup_pd(pnt_bp1++);
        vec_b_p2.vec = _mm_loaddup_pd(pnt_bp2++);
        vec_b_p3.vec = _mm_loaddup_pd(pnt_bp3++);
        // 乘加，好像没有重载+=运算符，自己拼...
        // C的一二行
        vec_c_00_10.vec = _mm_add_pd(vec_c_00_10.vec, _mm_mul_pd(vec_a_0p_1p.vec, vec_b_p0.vec));
        vec_c_01_11.vec = _mm_add_pd(vec_c_01_11.vec, _mm_mul_pd(vec_a_0p_1p.vec, vec_b_p1.vec));
        vec_c_02_12.vec = _mm_add_pd(vec_c_02_12.vec, _mm_mul_pd(vec_a_0p_1p.vec, vec_b_p2.vec));
        vec_c_03_13.vec = _mm_add_pd(vec_c_03_13.vec, _mm_mul_pd(vec_a_0p_1p.vec, vec_b_p3.vec));
        // C的三四行
        vec_c_20_30.vec = _mm_add_pd(vec_c_20_30.vec, _mm_mul_pd(vec_a_2p_3p.vec, vec_b_p0.vec));
        vec_c_21_31.vec = _mm_add_pd(vec_c_21_31.vec, _mm_mul_pd(vec_a_2p_3p.vec, vec_b_p1.vec));
        vec_c_22_32.vec = _mm_add_pd(vec_c_22_32.vec, _mm_mul_pd(vec_a_2p_3p.vec, vec_b_p2.vec));
        vec_c_23_33.vec = _mm_add_pd(vec_c_23_33.vec, _mm_mul_pd(vec_a_2p_3p.vec, vec_b_p3.vec));
    }
    // 之所以用union，是这里可以逐个取址，如C(0,0)+=vec_c_00_10.d[0]，这样，但是后面一想太麻烦了，于是直接store了
    _mm_store_pd(&C(0, 0), vec_c_00_10.vec);
    _mm_store_pd(&C(0, 1), vec_c_01_11.vec);
    _mm_store_pd(&C(0, 2), vec_c_02_12.vec);
    _mm_store_pd(&C(0, 3), vec_c_03_13.vec);

    _mm_store_pd(&C(2, 0), vec_c_20_30.vec);
    _mm_store_pd(&C(2, 1), vec_c_21_31.vec);
    _mm_store_pd(&C(2, 2), vec_c_22_32.vec);
    _mm_store_pd(&C(2, 3), vec_c_23_33.vec);
}
#define A(i, j) a[(i) + (j)*lda]
#define B(i, j) b[(i) + (j)*ldb]
#define C(i, j) c[(i) + (j)*ldc]

/**
 * 2. 采用intrinsic将值进行存储计算
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

#include "intrin.h"

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
        // a的初始化
        vec_a_0p_1p.vec = _mm_load_pd(&A(0, p));
        vec_a_2p_3p.vec = _mm_load_pd(&A(2, p));
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
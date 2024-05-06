#include <iostream>
#include <cstdlib>
#include <ctime>
#include <immintrin.h>
using namespace std;

// #ifdef __AVX__
// #include <immintrin.h>
// #else
// #warning xxx
// #endif

/**
 * 这个是给x这个数组加上对应的索引又写回x,n是数组x的size
 * n是4的倍数
 */
void addindex_vec2(double *x, int n)
{
    __m256d x_vec, incr, ind;
    ind = _mm256_set_pd(3, 2, 1, 0);
    incr = _mm256_set1_pd(4);
    for (int i = 0; i < n; i += 4)
    {
        x_vec = _mm256_load_pd(x + i);     // load 4 doubles
        x_vec = _mm256_add_pd(x_vec, ind); // add the two
        ind = _mm256_add_pd(ind, incr);    // update ind
        _mm256_store_pd(x + i, x_vec);     // store back
    }
}

int main(int argc, char const *argv[])
{
    srand(time(0));
    __m256 a = _mm256_set_ps(1.0, 2.1, 3.2, 4.3,
                             5.4, 6.5, 7.6, 8.7);

    __m256 b = _mm256_set_ps(3.8, 4.9, 6.0, 7.1,
                             8.2, 9.3, 10.4, 11.5);

    __m256 c = _mm256_add_ps(a, b);

    float d[8];

    _mm256_storeu_ps(d, c);

    for (int i = 0; i < 8; i++)
        cout << d[i] << ends;
    cout << endl;

    double *p = new double[32];
    for (int i = 0; i < 32; i++)
    {
        p[i] = rand() % 100;
        cout << p[i] << ends;
    }
    addindex_vec2(p, 32);

    cout << endl;
    for (int i = 0; i < 32; i++)
        cout << p[i] << ends;
    cout << endl;

    delete[] p;

    return 0;
}

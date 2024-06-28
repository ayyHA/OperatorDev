#include "gemm.h"
#include "assert.h"
#include "config.h"
#include <cstdlib>
#ifdef USE_NEON
    #include <arm_neon.h>
#endif // USE_NEON

#define A(i,j) a[(i)*lda + (j)]
#define B(i,j) b[(i)*ldb + (j)]
#define C(i,j) c[(i)*ldc + (j)]

// MKN == PQR 分块处理的MKN的size
// 经过计算后的数据,输出的txt为record_gemm_4x4_like_blas_asmV3
#define GEMM_M (1024)
#define GEMM_K (256)
#define GEMM_N (256)




// 每次的操作展开4,循环展开4的意思,因为micro_kernel里就是一次处理4个4x4的矩阵(重排后很好取数据,一次做掉k维上4个列(A)/4个行(B))
#define GEMM_UNROLL (4)


// A阵打包
/* 
 A(需要注意,内部想通过外积的方式来处理矩阵计算,一个for希望完成一个C阵4x4小块的计算,因此需要取A的一列(size为4)):
A11:         A12:
 1  2  3  4   5  6  7  8
 9 10 11 12  13 14 15 16
17 18 19 20  21 22 23 24
25 26 27 28  29 30 31 32
A21:         A22:
11 22 33 44  55 66 77 88
 9 10 11 12  13 14 15 16
17 18 19 20  21 22 23 24
25 26 27 28  29 30 31 32
----->
sa:
1 9 17 15 2 10 18 26 3 11 19 27 4 12 20 28 (A11)
5 13 21 29 6 14 22 30 7 15 23 31 8 16 24 32 (A12)
11 9 17 25 22 10 18 26 33 11 19 27 44 12 20 28(A21)
55 13 21 29 66 14 22 30 77 15 23 31 88 16 24 32(A22)
以上代表了A分块的部分重排为sa的内存排列顺序,其中A11,A12是连续排在一起的,别的亦是

显然以A11为例,我们计算是以:
[1 9 17 15] [2 10 18 26] [3 11 19 27] [4 12 20 28] (A11)
为一个float32x4_t来进行读取的
*/
void packA_4(int m,int k,float* a,int lda,float* sa);
// B阵打包
/*
 B(需要注意,内部是通过外积的方式完成计算,一次计算的目的是解决一个C阵的4x4阵,因此B取一行(size为4))
B11:          B12:
10 11 12 13   16 17 18 19   
14 15 16 17   20 21 22 23
18 19 20 21   24 25 26 27
22 23 24 25   28 29 30 31
B21:          B22:
32 33 34 35   48 49 50 51
36 37 38 39   52 53 54 55
40 41 42 43   56 57 58 59
44 45 46 47   60 61 62 63
----->
sb:
10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25(B11)
32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47(B21)
16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31(B12)
48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63(B22)
以上代表了B分块的部分重排后的内存排序,B11和B21是连着的,另外的同理

显然以B11为例,我们计算是以:
[10 11 12 13] [14 15 16 17] [18 19 20 21] [22 23 24 25] (B11)
为一个float32x4_t来进行读取的
 */
void packB_4(int k,int n,float* b,int ldb,float* sb);

// 微内核操作,真正实现矩阵乘的函数
void kernel_4x4_V2(int m,int n,int k,float* a,float* b,float* c,int ldc);


// 64字节对齐的地址
namespace asmV3{
    float* cacheMalloc(int size){
        void* address = nullptr;
        int flag = posix_memalign(&address,64,size*sizeof(float));
        assert(flag == 0);
        return (float*)address;
    }
}
/*
    入口函数
    需要注意,目前只能处理4的倍数的矩阵size
 */
void gemm_4x4_like_blas_asmV3(int m,int n,int k,float* a,int lda,
                            float* b,int ldb,float* c,int ldc){
    assert(m%4==0 && n%4==0 && k%4==0);
    assert(m>0 && n>0 && k>0);

    int min_m,min_n,min_k;  // m,n,k的分块大小
    int ms,ns,ks;           // m,n,k分块的start位置
    int min_mm;             // m再分块的大小
    int mms;                // m再分块的start位置

    float* sa,*sb;
    sa = asmV3::cacheMalloc(m*k);
    sb = asmV3::cacheMalloc(k*n);
    
    for(ms=0;ms<m;ms+=GEMM_M){  // 粗略切个M,感觉像是A充满L2Cache?
        min_m = m-ms;
        if(min_m>GEMM_M){
            min_m = GEMM_M;
        }

        for(ks=0;ks<k;ks+=min_k){   // 切K,给我的感觉是每次都很均匀
            min_k = k-ks;
            if(min_k >= 2*GEMM_K){
                min_k = GEMM_K;
            }
            else if(min_k > GEMM_K){
                // 这里分的均匀且保证这一次是切成4的倍数
                min_k = (min_k/2 + GEMM_UNROLL -1) &~(GEMM_UNROLL -1);
            }

            // 对B第一次打包前的预处理
            min_n = n;
            if(min_n >= 2*GEMM_N){
                min_n = GEMM_N;
            }
            else if(min_n > GEMM_N){
                min_n = (min_n/2 + GEMM_UNROLL - 1)&~(GEMM_UNROLL -1);
            }

            // B的第一次打包
            packB_4(min_k,min_n,b+ks*ldb,ldb,sb);

            for(mms=ms; mms<ms+min_m; mms+=min_mm){
                min_mm = (ms+min_m) - mms;
                if(min_mm >= 3*GEMM_UNROLL){
                    min_mm = 3*GEMM_UNROLL;
                }
                else if(min_mm >= 2*GEMM_UNROLL){
                    min_mm = 2*GEMM_UNROLL;
                }
                else if(min_mm >= GEMM_UNROLL){
                    min_mm = GEMM_UNROLL;
                }

                // 依次打包A
                packA_4(min_mm,min_k,a+mms*lda+ks,lda,sa+(mms-ms)*min_k);
                // 计算此时的C
                kernel_4x4_V2(min_mm,min_n,min_k,sa+(mms-ms)*min_k,sb,c+mms*ldc,ldc);
            }

            // 同一ks下,B的剩余列数依次打包
            for(ns=min_n;ns<n;ns+=min_n){
                min_n = n-ns;
                if(min_n >= 2*GEMM_N){
                    min_n = GEMM_N;
                }else if(min_n > GEMM_N){
                    min_n = (min_n/2 + GEMM_UNROLL -1) &~ (GEMM_UNROLL-1);
                }
                packB_4(min_k,min_n,b+ks*ldb+ns,ldb,sb);
                kernel_4x4_V2(min_m,min_n,min_k,sa,sb,c+ms*ldc+ns,ldc);
            }
        }
    }

    free(sb);
    free(sa);
}

void packA_4(int m,int k,float* from,int lda,float* to){
    assert( m>0 && k>0 && m%4==0 && k%4==0);
    
    float* a = from;
    float* packedA = to;
    float tmp1,tmp2,tmp3,tmp4,
          tmp5,tmp6,tmp7,tmp8,
          tmp9,tmp10,tmp11,tmp12,
          tmp13,tmp14,tmp15,tmp16;

    int i;
    for(i=0;i+3<m;i+=4){
        float* a1,*a2,*a3,*a4;
        a1 = a;
        a2 = a + lda;
        a3 = a + lda*2;
        a4 = a + lda*3;

        int j;
        for(j=0;j+3<k;j+=4){
            tmp1 = *a1;
            tmp2 = *a2;
            tmp3 = *a3;
            tmp4 = *a4;

            tmp5 = *(a1+1);
            tmp6 = *(a2+1);
            tmp7 = *(a3+1);
            tmp8 = *(a4+1);
        
            tmp9 = *(a1+2);
            tmp10 = *(a2+2);
            tmp11 = *(a3+2);
            tmp12 = *(a4+2);

            tmp13 = *(a1+3);
            tmp14 = *(a2+3);
            tmp15 = *(a3+3);
            tmp16 = *(a4+3);

            *(packedA) = tmp1;
            *(packedA+1) = tmp2;
            *(packedA+2) = tmp3;
            *(packedA+3) = tmp4;

            *(packedA+4) = tmp5;
            *(packedA+5) = tmp6;
            *(packedA+6) = tmp7;
            *(packedA+7) = tmp8;

            *(packedA+8) = tmp9;
            *(packedA+9) = tmp10;
            *(packedA+10) = tmp11;
            *(packedA+11) = tmp12;

            *(packedA+12) = tmp13;
            *(packedA+13) = tmp14;
            *(packedA+14) = tmp15;
            *(packedA+15) = tmp16;

            a1+=4;
            a2+=4;
            a3+=4;
            a4+=4;

            packedA+=16;
        }
        a+=4*lda;  
    }
}

void packB_4(int k,int n,float* from,int ldb,float* to){
    float* b = from;
    float* packedB;
    float tmp1,tmp2,tmp3,tmp4,
          tmp5,tmp6,tmp7,tmp8,
          tmp9,tmp10,tmp11,tmp12,
          tmp13,tmp14,tmp15,tmp16;

    int i;
    for(i=0;i+3<k;i+=4){    // 处理4行
        packedB = to + i*4;
        float* b1 = b;
        float* b2 = b+ldb;
        float* b3 = b+ldb*2;
        float* b4 = b+ldb*3;

        int j;
        for(j=0;j+3<n;j+=4){    // 处理4列
            tmp1 = *(b1);
            tmp2 = *(b1+1);
            tmp3 = *(b1+2);
            tmp4 = *(b1+3);

            tmp5 = *(b2);
            tmp6 = *(b2+1);
            tmp7 = *(b2+2);
            tmp8 = *(b2+3);

            tmp9 = *(b3);
            tmp10 = *(b3+1);
            tmp11 = *(b3+2);
            tmp12 = *(b3+3);

            tmp13 = *(b4);
            tmp14 = *(b4+1);
            tmp15 = *(b4+2);
            tmp16 = *(b4+3);

            *packedB = tmp1;
            *(packedB+1) = tmp2;
            *(packedB+2) = tmp3;
            *(packedB+3) = tmp4;

            *(packedB+4) = tmp5;
            *(packedB+5) = tmp6;
            *(packedB+6) = tmp7;
            *(packedB+7) = tmp8;

            *(packedB+8) = tmp9;
            *(packedB+9) = tmp10;
            *(packedB+10) = tmp11;
            *(packedB+11) = tmp12;

            *(packedB+12) = tmp13;
            *(packedB+13) = tmp14;
            *(packedB+14) = tmp15;
            *(packedB+15) = tmp16;
            
            // *packedB = *(b1);
            // *(packedB+1) = *(b1+1);
            // *(packedB+2) = *(b1+2); 
            // *(packedB+3) = *(b1+3);
        
            // *(packedB+4) = *(b2);
            // *(packedB+5) = *(b2+1);        
            // *(packedB+6) = *(b2+2);        
            // *(packedB+7) = *(b2+3);        
        
            // *(packedB+8) = *(b3);
            // *(packedB+9) = *(b3+1);        
            // *(packedB+10) = *(b3+2);        
            // *(packedB+11) = *(b3+3);        
        
            // *(packedB+12) = *(b4);        
            // *(packedB+13) = *(b4+1);        
            // *(packedB+14) = *(b4+2);        
            // *(packedB+15) = *(b4+3);        

            b1+=4;
            b2+=4;
            b3+=4;
            b4+=4;
            
            packedB += k*4;
        }
        b+=4*ldb;
    }
}

void kernel_4x4_V2(int m,int n,int k,float* a,float* b,float* c,int ldc){
    
    int i,j,p;
    float* sa = a,*sb = b,*sc = c;
#ifdef __aarch64__
    int ldc_offset = ldc*sizeof(float);
#endif

    // 一次处理一个4x4阵,内部的k一次移动4次,即完成4个4x4阵(通过外积)的计算和累和
    for(i=0;i+3<m;i+=4){
        for(j=0;j+3<n;j+=4){
        #ifdef __aarch64__
                __asm__ volatile(
                /* 初始化,Caux {v0 - v3},直接装C0-C3 */
                "ld1 {v0.4s},[%[sc]]    \n"
                "add x0,%[sc],%[ldc_offset] \n"
                "ld1 {v1.4s},[x0]  \n"
                "add x1,x0,%[ldc_offset]    \n"
                "ld1 {v2.4s},[x1]   \n"
                "add x2,x1,%[ldc_offset]    \n"
                "ld1 {v3.4s},[x2]   \n"
                "asr x9,%[k],#2 \n"
                "loop:  \n"
                "   prfm pldl1keep,[%[sa],#256] \n"
                "   prfm pldl1keep,[%[sb],#256] \n"
                "   ld1 {v4.4s},[%[sa]],#16  \n"
                "   ld1 {v5.4s},[%[sb]],#16  \n"
                "   fmla v0.4s,v5.4s,v4.s[0]    \n"
                "   fmla v1.4s,v5.4s,v4.s[1]    \n"
                "   fmla v2.4s,v5.4s,v4.s[2]    \n"
                "   fmla v3.4s,v5.4s,v4.s[3]    \n"
                "   ld1 {v6.4s},[%[sa]],#16  \n"
                "   ld1 {v7.4s},[%[sb]],#16  \n"
                "   fmla v0.4s,v7.4s,v6.s[0]    \n"
                "   fmla v1.4s,v7.4s,v6.s[1]    \n"
                "   fmla v2.4s,v7.4s,v6.s[2]    \n"
                "   fmla v3.4s,v7.4s,v6.s[3]    \n"
                "   ld1 {v4.4s},[%[sa]],#16  \n"
                "   ld1 {v5.4s},[%[sb]],#16  \n"
                "   fmla v0.4s,v5.4s,v4.s[0]    \n"
                "   fmla v1.4s,v5.4s,v4.s[1]    \n"
                "   fmla v2.4s,v5.4s,v4.s[2]    \n"
                "   fmla v3.4s,v5.4s,v4.s[3]    \n"
                "   ld1 {v6.4s},[%[sa]],#16 \n"
                "   ld1 {v7.4s},[%[sb]],#16 \n"
                "   fmla v0.4s,v7.4s,v6.s[0]    \n"
                "   fmla v1.4s,v7.4s,v6.s[1]    \n"
                "   fmla v2.4s,v7.4s,v6.s[2]    \n"
                "   fmla v3.4s,v7.4s,v6.s[3]    \n"
                "   subs x9,x9,#1   \n"
                "   bne loop    \n"
                // SAVE {v0 - v3}
                "st1 {v0.4s},[%[sc]]    \n"
                "st1 {v1.4s},[x0]   \n"
                "st1 {v2.4s},[x1]   \n"
                "st1 {v3.4s},[x2]   \n"
                : [sa]  "+r"(sa),
                  [sb]  "+r"(sb),
                  [sc]  "+r"(sc)
                : [k]   "r"(k),
                  [ldc_offset]  "r"(ldc_offset)
                : "cc","memory","x0","x1","x2","x9","v0","v1","v2","v3","v4","v5","v6","v7"
                );

        #else
            float32x4_t C0 = {0},C1 = {0},C2 = {0},C3 ={0};
            __builtin_prefetch(sa,0,3);
            __builtin_prefetch(sb,0,3);

            for(p=0;p+3<k;p+=4){

                float32x4_t A1_ip = vld1q_f32(sa);
                float32x4_t B1_pj = vld1q_f32(sb);

                C0 = vmlaq_laneq_f32(C0,B1_pj,A1_ip,0);
                C1 = vmlaq_laneq_f32(C1,B1_pj,A1_ip,1);
                C2 = vmlaq_laneq_f32(C2,B1_pj,A1_ip,2);
                C3 = vmlaq_laneq_f32(C3,B1_pj,A1_ip,3);

                float32x4_t A2_ip = vld1q_f32(sa+4);
                float32x4_t B2_pj = vld1q_f32(sb+4);

                C0 = vmlaq_laneq_f32(C0,B2_pj,A2_ip,0);
                C1 = vmlaq_laneq_f32(C1,B2_pj,A2_ip,1);
                C2 = vmlaq_laneq_f32(C2,B2_pj,A2_ip,2);
                C3 = vmlaq_laneq_f32(C3,B2_pj,A2_ip,3);

                float32x4_t A3_ip = vld1q_f32(sa+8);
                float32x4_t B3_pj = vld1q_f32(sb+8);

                C0 = vmlaq_laneq_f32(C0,B3_pj,A3_ip,0);
                C1 = vmlaq_laneq_f32(C1,B3_pj,A3_ip,1);
                C2 = vmlaq_laneq_f32(C2,B3_pj,A3_ip,2);
                C3 = vmlaq_laneq_f32(C3,B3_pj,A3_ip,3);

                float32x4_t A4_ip = vld1q_f32(sa+12);
                float32x4_t B4_pj = vld1q_f32(sb+12);

                C0 = vmlaq_laneq_f32(C0,B4_pj,A4_ip,0);
                C1 = vmlaq_laneq_f32(C1,B4_pj,A4_ip,1);
                C2 = vmlaq_laneq_f32(C2,B4_pj,A4_ip,2);
                C3 = vmlaq_laneq_f32(C3,B4_pj,A4_ip,3);
                
                __builtin_prefetch(sa+16,0,3);
                __builtin_prefetch(sb+16,0,3);

                sa+=16;
                sb+=16;
            }           // endp

            float32x4_t c00 = vld1q_f32(sc);
            float32x4_t c10 = vld1q_f32(sc+ldc);
            float32x4_t c20 = vld1q_f32(sc+2*ldc);
            float32x4_t c30 = vld1q_f32(sc+3*ldc);

            c00 = vaddq_f32(c00,C0);
            c10 = vaddq_f32(c10,C1);
            c20 = vaddq_f32(c20,C2);
            c30 = vaddq_f32(c30,C3);

            vst1q_f32(sc,c00);
            vst1q_f32(sc+ldc,c10);
            vst1q_f32(sc+ldc*2,c20);
            vst1q_f32(sc+ldc*3,c30);
        
        #endif // __aarch64__
            sa-=4*k;    // A指针退回原位
            sc+=4;
        }               // endj
        sb = b;         // B指针退回原位
        sa += 4*k;      // A指针增加到对应位置
        c += 4*ldc;
        sc = c;
    }                   // endi
}
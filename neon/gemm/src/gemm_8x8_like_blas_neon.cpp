#include "gemm.h"
#include "config.h"
#include "assert.h"
#include <cstdlib>
#ifdef USE_NEON
    #include <arm_neon.h>
#endif  // USE_NEON

/*
    在前面的部分,打包的时候我们其实也可以向量化处理:
    packA_4(4行的首个元素)其中有m,k,遍历k的部分也设置为4其实就可以向量化处理
    packB_4(4列的每一列)其中有k,n,遍历n的部分设置为4(这里packB_4说明就是4)也可以向量化处理
    打包方式决定了微内核的计算方式,这里A是4,B是4,则微内核是4x4;如果是packA_8,packB_12,则微内核是8x12
    
    unroll是否可以展开更多,比如8,比如12?这个是指最终微内核里的k的那部分
    
    根据ARMV82的架构,展开:微内核中的mrxnr是否可以更大,比如8x8
*/

#define A(i,j) a[(i)*(lda) + (j)]
#define B(i,j) b[(i)*(ldb) + (j)]
#define C(i,j) c[(i)*(ldc) + (j)]

#define GEMM_M (1024)
#define GEMM_K (256)
#define GEMM_N (256)

#define GEMM_UNROLL_8 (8)
#define GEMM_UNROLL_4 (4)

void packA_8(int m,int k,float* src,int lda,float* dst);
void packB_8(int k,int n,float* src,int ldb,float* dst);
void kernel8x8(int m,int n,int k,float* a,float* b,float* c,int ldc);

// cacheline对齐的malloc内存
float* cacheMalloc(int size){
    int nBytes = sizeof(float)*size;
    void* src = nullptr;
    int status = posix_memalign(&src,64,nBytes);
    assert(status==0);
    return (float*)src;
}

/* 本例仅支持m,n,k为8的倍数的运算,对于边界暂无作额外处理 */
void gemm_8x8_like_blas_neon(int m,int n,int k,float* a,int lda,float* b,int ldb,float*c,int ldc){
    assert(m>0 && n>0 && k>0 && m%8==0 && n%8==0 && k%8==0);
    // 新的疑惑sa,sb有必要开这么大吗?开GEMM_M*GEMM_K,GEMM_K*GEMM_N大小不就好了吗
    float* sa = cacheMalloc(m*k);
    float* sb = cacheMalloc(k*n);

    int mstart,nstart,kstart,mmstart;
    int min_m,min_n,min_k,min_mm;

    // M分作小块,以GEMM_M为基本尺度
    for(mstart=0;mstart<m;mstart+=GEMM_M){
        min_m = m-mstart;
        if(min_m>=GEMM_M){
            min_m = GEMM_M;
        }
    
        // 正式开切,切K,k内部unroll的尺度为4,所以期望除了最后一次外,都要是4的倍数
        for(kstart=0;kstart<k;kstart+=min_k){
            min_k = k-kstart;
            if(min_k >= 2*GEMM_K){
                min_k = GEMM_K;           
            }
            else if(min_k > GEMM_K){
                min_k = (min_k/2 + GEMM_UNROLL_4 -1) &~ (GEMM_UNROLL_4 -1);
            }

            // 首先处理一个开头的B,并对它作打包
            min_n = n;
            if(min_n > 2*GEMM_N){
                min_n = GEMM_N;
            }
            else if(min_n > GEMM_N){
                min_n = (min_n/2 + GEMM_UNROLL_8-1) &~ (GEMM_UNROLL_8 -1);   
            }
            packB_8(min_k,min_n,b+kstart*ldb,ldb,sb);
            
            // 切[mstart,mstart+min_m)这一块区域的m,并打包,然后与打包好的上述B块计算.边打包边计算
            for(mmstart=mstart;mmstart<mstart+min_m;mmstart+=min_mm){
                min_mm = (mstart+min_m) - mmstart;
                if(min_mm >= 3*GEMM_UNROLL_8){
                    min_mm = 3*GEMM_UNROLL_8;
                }
                else if(min_mm >= 2*GEMM_UNROLL_8){
                    min_mm = 2*GEMM_UNROLL_8;
                }
                else if(min_mm >= GEMM_UNROLL_8){
                    min_mm = GEMM_UNROLL_8; 
                }
                
                packA_8(min_mm,min_k,a+(mmstart)*lda+kstart,lda,sa+(mmstart-mstart)*min_k);
                kernel8x8(min_mm,min_n,min_k,sa+(mmstart-mstart)*min_k,sb,c+mmstart*ldc,ldc);
            }

            // 正式切n,把剩余的n切作min_n大小的块,从nstart开始,通过迭代直至完成[kstart,kstart+min_k)下的B block与A block的计算
            for(nstart=min_n;nstart<n;nstart+=min_n){
                min_n = n-nstart;
                if(min_n >= 2*GEMM_N){
                    min_n = GEMM_N;
                }
                else if(min_n > GEMM_N){
                    min_n = (min_n/2 + GEMM_UNROLL_8 -1 ) &~ (GEMM_UNROLL_8-1);
                }

                packB_8(min_k,min_n,b+kstart*ldb+nstart,ldb,sb);
                kernel8x8(min_m,min_n,min_k,sa,sb,c+mstart*ldc+nstart,ldc);
            }
        }
    
    }

    free(sb);
    free(sa);
}


/*
packA:
0 1 2 3  4 5 6 7   7 8 9 a  b c d e 
8 9 a b  c d e f   f 0 1 2  3 4 5 6 
0 1 2 3  4 5 6 7   7 8 9 a  b c d e
8 9 a b  c d e f   f 0 1 2  3 4 5 6
0 1 2 3  4 5 6 7   7 8 9 a  b c d e
8 9 a b  c d e f   f 0 1 2  3 4 5 6
0 1 2 3  4 5 6 7   7 8 9 a  b c d e
8 9 a b  c d e f   f 0 1 2  3 4 5 6

->  0 8 2 a -> 0 8 0 8 ->  0 8 0 8 0 8 0 8
    1 9 3 b    1 9 1 9     ...
    0 8 2 a    2 a 2 a
    1 9 3 b    3 b 3 b
    0 8 2 a    0 8 0 8     
    1 9 3 b    1 9 1 9     
    0 8 2 a    2 a 2 a  
    1 9 3 b    3 b 3 b

0 8 0 8 0 8 0 8
1 9 1 9 1 9 1 9
2 a 2 a 2 a 2 a
3 b 3 b 3 b 3 b

4 c 4 c 4 c 4 c
5 d 5 d 5 d 5 d
6 e 6 e 6 e 6 e  
7 f 7 f 7 f 7 f
...
*/
void packA_8(int m,int k,float* src,int lda,float* dst){
    assert(m>0 && k>0 && m%8==0 && k%4==0);

    float *srcA,*dstA;
    int i,j;
    float *tmpA0,*tmpA1,*tmpA2,*tmpA3,*tmpA4,*tmpA5,*tmpA6,*tmpA7;
    
    dstA = dst;
    for(i=0;i+7<m;i+=8){
        srcA = src+i*lda;
        tmpA0 = srcA;
        tmpA1 = srcA + lda;
        tmpA2 = srcA + lda*2;
        tmpA3 = srcA + lda*3;
        tmpA4 = srcA + lda*4;
        tmpA5 = srcA + lda*5;
        tmpA6 = srcA + lda*6;
        tmpA7 = srcA + lda*7;
        
        for(j=0;j+3<k;j+=4){
            float32x4_t v0 = vld1q_f32(tmpA0);
            float32x4_t v1 = vld1q_f32(tmpA1);
            float32x4_t v2 = vld1q_f32(tmpA2);
            float32x4_t v3 = vld1q_f32(tmpA3);
            float32x4_t v4 = vld1q_f32(tmpA4);
            float32x4_t v5 = vld1q_f32(tmpA5);
            float32x4_t v6 = vld1q_f32(tmpA6);
            float32x4_t v7 = vld1q_f32(tmpA7);
            
            tmpA0+=4;
            tmpA1+=4;
            tmpA2+=4;
            tmpA3+=4;
            tmpA4+=4;
            tmpA5+=4;
            tmpA6+=4;
            tmpA7+=4;

            float32x4x2_t v01 = vtrnq_f32(v0,v1);
            float32x4x2_t v23 = vtrnq_f32(v2,v3);
            float32x4x2_t v45 = vtrnq_f32(v4,v5);
            float32x4x2_t v67 = vtrnq_f32(v6,v7);

            float32x4_t _v0 = vcombine_f32(vget_low_f32(v01.val[0]),vget_low_f32(v23.val[0]));
            float32x4_t _v1 = vcombine_f32(vget_low_f32(v01.val[1]),vget_low_f32(v23.val[1]));
            float32x4_t _v2 = vcombine_f32(vget_high_f32(v01.val[0]),vget_high_f32(v23.val[0]));
            float32x4_t _v3 = vcombine_f32(vget_high_f32(v01.val[1]),vget_high_f32(v23.val[1]));

            float32x4_t _v4 = vcombine_f32(vget_low_f32(v45.val[0]),vget_low_f32(v67.val[0]));
            float32x4_t _v5 = vcombine_f32(vget_low_f32(v45.val[1]),vget_low_f32(v67.val[1]));
            float32x4_t _v6 = vcombine_f32(vget_high_f32(v45.val[0]),vget_high_f32(v67.val[0]));
            float32x4_t _v7 = vcombine_f32(vget_high_f32(v45.val[1]),vget_high_f32(v67.val[1]));

            vst1q_f32(dstA,_v0);
            vst1q_f32(dstA+4,_v4);
            vst1q_f32(dstA+8,_v1);
            vst1q_f32(dstA+12,_v5);
            vst1q_f32(dstA+16,_v2);
            vst1q_f32(dstA+20,_v6);
            vst1q_f32(dstA+24,_v3);
            vst1q_f32(dstA+28,_v7);
           
            dstA+=32;
        }
    }
}

/*
packB:
0 1 2 3  4 5 6 7   7 8 9 a  b c d e 
8 9 a b  c d e f   f 0 1 2  3 4 5 6 
0 1 2 3  4 5 6 7   7 8 9 a  b c d e
8 9 a b  c d e f   f 0 1 2  3 4 5 6
0 1 2 3  4 5 6 7   7 8 9 a  b c d e
8 9 a b  c d e f   f 0 1 2  3 4 5 6
0 1 2 3  4 5 6 7   7 8 9 a  b c d e
8 9 a b  c d e f   f 0 1 2  3 4 5 6

0 1 2 3 4 5 6 7 
8 9 a b c d e f  | 
                 |
                 |
                 8*k
                 |
                 |
...              |
7 8 9 a b c d e
f 0 1 2 3 4 5 6   
... 
 */
void packB_8(int k,int n,float* src,int ldb,float* dst){
    // assert(k>0 && n>0 && k%4==0 && n%8==0);

    float* srcB;
    float* dstB;

    int i,j;
    for(i=0;i<k;i++){
        srcB = src + i*ldb;
        dstB = dst + i*8;   // 可以将其想象成是行主序,且leading dimension为8的矩阵
        for(j=0;j+7<n;j+=8){
            float32x4_t v0 = vld1q_f32(srcB);
            float32x4_t v1 = vld1q_f32(srcB+4);
            srcB+=8;
            vst1q_f32(dstB,v0);
            vst1q_f32(dstB+4,v1);
            dstB+=8*k;
        }
    }
}

// micro-kernel
void kernel8x8(int m,int n,int k,float* a,float* b,float* c,int ldc){
    assert(k>0 && k%4==0);  // k_unroll==4
    
    int i,j,p;
    float* tmpA=a,*tmpB=b,*tmpC=c;
    for(i=0;i+7<m;i+=8){
        for(j=0;j+7<n;j+=8){
            __builtin_prefetch(tmpA,0,3);
            __builtin_prefetch(tmpB,0,3);
            float32x4_t C00={0},C01={0},C10={0},C11={0},C20={0},C21={0},C30={0},C31={0},
                        C40={0},C41={0},C50={0},C51={0},C60={0},C61={0},C70={0},C71={0};
            
            for(p=0;p+3<k;p+=4){
                float32x4_t v00A = vld1q_f32(tmpA);
                float32x4_t v10A = vld1q_f32(tmpA+4);
                tmpA+=8;
                float32x4_t v00B = vld1q_f32(tmpB);
                float32x4_t v01B = vld1q_f32(tmpB+4);
                tmpB+=8;
                C00 = vmlaq_laneq_f32(C00,v00B,v00A,0);
                C10 = vmlaq_laneq_f32(C10,v00B,v00A,1);
                C20 = vmlaq_laneq_f32(C20,v00B,v00A,2);
                C30 = vmlaq_laneq_f32(C30,v00B,v00A,3);
                
                C01 = vmlaq_laneq_f32(C01,v01B,v00A,0);
                C11 = vmlaq_laneq_f32(C11,v01B,v00A,1);
                C21 = vmlaq_laneq_f32(C21,v01B,v00A,2);
                C31 = vmlaq_laneq_f32(C31,v01B,v00A,3);

                C40 = vmlaq_laneq_f32(C40,v00B,v10A,0);
                C50 = vmlaq_laneq_f32(C50,v00B,v10A,1);
                C60 = vmlaq_laneq_f32(C60,v00B,v10A,2);
                C70 = vmlaq_laneq_f32(C70,v00B,v10A,3);

                C41 = vmlaq_laneq_f32(C41,v01B,v10A,0);
                C51 = vmlaq_laneq_f32(C51,v01B,v10A,1);
                C61 = vmlaq_laneq_f32(C61,v01B,v10A,2);
                C71 = vmlaq_laneq_f32(C71,v01B,v10A,3);

                v00A = vld1q_f32(tmpA);
                v10A = vld1q_f32(tmpA+4);
                tmpA+=8;

                v00B = vld1q_f32(tmpB);
                v01B = vld1q_f32(tmpB+4);
                tmpB+=8;

                C00 = vmlaq_laneq_f32(C00,v00B,v00A,0);
                C10 = vmlaq_laneq_f32(C10,v00B,v00A,1);
                C20 = vmlaq_laneq_f32(C20,v00B,v00A,2);
                C30 = vmlaq_laneq_f32(C30,v00B,v00A,3);
                
                C01 = vmlaq_laneq_f32(C01,v01B,v00A,0);
                C11 = vmlaq_laneq_f32(C11,v01B,v00A,1);
                C21 = vmlaq_laneq_f32(C21,v01B,v00A,2);
                C31 = vmlaq_laneq_f32(C31,v01B,v00A,3);

                C40 = vmlaq_laneq_f32(C40,v00B,v10A,0);
                C50 = vmlaq_laneq_f32(C50,v00B,v10A,1);
                C60 = vmlaq_laneq_f32(C60,v00B,v10A,2);
                C70 = vmlaq_laneq_f32(C70,v00B,v10A,3);

                C41 = vmlaq_laneq_f32(C41,v01B,v10A,0);
                C51 = vmlaq_laneq_f32(C51,v01B,v10A,1);
                C61 = vmlaq_laneq_f32(C61,v01B,v10A,2);
                C71 = vmlaq_laneq_f32(C71,v01B,v10A,3);

                v00A = vld1q_f32(tmpA);
                v10A = vld1q_f32(tmpA+4);
                tmpA+=8;

                v00B = vld1q_f32(tmpB);
                v01B = vld1q_f32(tmpB+4);
                tmpB+=8;

                C00 = vmlaq_laneq_f32(C00,v00B,v00A,0);
                C10 = vmlaq_laneq_f32(C10,v00B,v00A,1);
                C20 = vmlaq_laneq_f32(C20,v00B,v00A,2);
                C30 = vmlaq_laneq_f32(C30,v00B,v00A,3);
                
                C01 = vmlaq_laneq_f32(C01,v01B,v00A,0);
                C11 = vmlaq_laneq_f32(C11,v01B,v00A,1);
                C21 = vmlaq_laneq_f32(C21,v01B,v00A,2);
                C31 = vmlaq_laneq_f32(C31,v01B,v00A,3);

                C40 = vmlaq_laneq_f32(C40,v00B,v10A,0);
                C50 = vmlaq_laneq_f32(C50,v00B,v10A,1);
                C60 = vmlaq_laneq_f32(C60,v00B,v10A,2);
                C70 = vmlaq_laneq_f32(C70,v00B,v10A,3);

                C41 = vmlaq_laneq_f32(C41,v01B,v10A,0);
                C51 = vmlaq_laneq_f32(C51,v01B,v10A,1);
                C61 = vmlaq_laneq_f32(C61,v01B,v10A,2);
                C71 = vmlaq_laneq_f32(C71,v01B,v10A,3);

                v00A = vld1q_f32(tmpA);
                v10A = vld1q_f32(tmpA+4);
                tmpA+=8;

                v00B = vld1q_f32(tmpB);
                v01B = vld1q_f32(tmpB+4);
                tmpB+=8;

                C00 = vmlaq_laneq_f32(C00,v00B,v00A,0);
                C10 = vmlaq_laneq_f32(C10,v00B,v00A,1);
                C20 = vmlaq_laneq_f32(C20,v00B,v00A,2);
                C30 = vmlaq_laneq_f32(C30,v00B,v00A,3);
                
                C01 = vmlaq_laneq_f32(C01,v01B,v00A,0);
                C11 = vmlaq_laneq_f32(C11,v01B,v00A,1);
                C21 = vmlaq_laneq_f32(C21,v01B,v00A,2);
                C31 = vmlaq_laneq_f32(C31,v01B,v00A,3);

                C40 = vmlaq_laneq_f32(C40,v00B,v10A,0);
                C50 = vmlaq_laneq_f32(C50,v00B,v10A,1);
                C60 = vmlaq_laneq_f32(C60,v00B,v10A,2);
                C70 = vmlaq_laneq_f32(C70,v00B,v10A,3);

                C41 = vmlaq_laneq_f32(C41,v01B,v10A,0);
                C51 = vmlaq_laneq_f32(C51,v01B,v10A,1);
                C61 = vmlaq_laneq_f32(C61,v01B,v10A,2);
                C71 = vmlaq_laneq_f32(C71,v01B,v10A,3);
            }
            vst1q_f32(tmpC,vaddq_f32(C00,vld1q_f32(tmpC)));
            vst1q_f32(tmpC+4,vaddq_f32(C01,vld1q_f32(tmpC+4)));
            vst1q_f32(tmpC+ldc,vaddq_f32(C10,vld1q_f32(tmpC+ldc)));
            vst1q_f32(tmpC+ldc+4,vaddq_f32(C11,vld1q_f32(tmpC+ldc+4)));
            vst1q_f32(tmpC+ldc*2,vaddq_f32(C20,vld1q_f32(tmpC+ldc*2)));
            vst1q_f32(tmpC+ldc*2+4,vaddq_f32(C21,vld1q_f32(tmpC+ldc*2+4)));
            vst1q_f32(tmpC+ldc*3,vaddq_f32(C30,vld1q_f32(tmpC+ldc*3)));
            vst1q_f32(tmpC+ldc*3+4,vaddq_f32(C31,vld1q_f32(tmpC+ldc*3+4)));

            vst1q_f32(tmpC+ldc*4,vaddq_f32(C40,vld1q_f32(tmpC+ldc*4)));
            vst1q_f32(tmpC+ldc*4+4,vaddq_f32(C41,vld1q_f32(tmpC+ldc*4+4)));
            vst1q_f32(tmpC+ldc*5,vaddq_f32(C50,vld1q_f32(tmpC+ldc*5)));
            vst1q_f32(tmpC+ldc*5+4,vaddq_f32(C51,vld1q_f32(tmpC+ldc*5+4)));
            vst1q_f32(tmpC+ldc*6,vaddq_f32(C60,vld1q_f32(tmpC+ldc*6)));
            vst1q_f32(tmpC+ldc*6+4,vaddq_f32(C61,vld1q_f32(tmpC+ldc*6+4)));
            vst1q_f32(tmpC+ldc*7,vaddq_f32(C70,vld1q_f32(tmpC+ldc*7)));
            vst1q_f32(tmpC+ldc*7+4,vaddq_f32(C71,vld1q_f32(tmpC+ldc*7+4)));

            tmpA -=8*k;
            tmpC +=8;
        }
        tmpB = b;
        c+=ldc*8;
        tmpC = c;
        tmpA += 8*k;
    }
}
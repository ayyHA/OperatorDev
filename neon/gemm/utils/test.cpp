#include "gemm.h"
#include "config.h"
#include <iostream>
#include <ctime>
#include <algorithm>
#include <iomanip>
#include "utils.h"
// extern "C" void random_matrix(int,int,float*,int);
// extern "C" void copy_matrix(int,int,float*,int,float*,int);
// extern "C" double compare_matrix(int,int,float*,int,float*,int);
// extern "C" double get_time(struct timespec*,struct timespec*);
/*
    行主序矩阵型性能测试分析
*/
int main(){
    // bool flag=true;
    int m,n,k;
    int lda,ldb,ldc;
    int rp;
    struct timespec start,end;
    double time_used;
    double best_time=1e6;
    double diff;
    double GFLOPs;
    float*a,*b,*c,*nowc,*keepc;
    for(int p=PSTART;p<=PEND;p+=PINC){
        best_time = 1e2;
        m = M==-1 ? p : M;
        n = N==-1 ? p : N;
        k = K==-1 ? p : K;
        GFLOPs = 2.0*m*n*k*(1e-9);
        lda = LDA==-1 ? k : LDA;
        ldb = LDB==-1 ? n : LDB;
        ldc = LDC==-1 ? n : LDC;

        a = new float[m*k];
        b = new float[k*n];
        c = new float[m*n];
        nowc = new float[m*n];   // origin计算后放这里
        keepc = new float[m*n];  // keepc存个档

        random_matrix(m,k,a,lda);
        random_matrix(k,n,b,ldb);
        random_matrix(m,n,nowc,ldc);

        copy_matrix(m,n,nowc,ldc,keepc,ldc);
        gemm_origin(m,n,k,a,lda,b,ldb,nowc,ldc);

        for(rp=0;rp<nRepeats;rp++){
            copy_matrix(m,n,keepc,ldc,c,ldc);
            
            clock_gettime(CLOCK_MONOTONIC_RAW,&start);
            // gemm_origin(m,n,k,a,lda,b,ldb,c,ldc);
            // gemm_4x1_origin(m,n,k,a,lda,b,ldb,c,ldc);
            // gemm_4x1_reg(m,n,k,a,lda,b,ldb,c,ldc);
            // gemm_4x4_reg(m,n,k,a,lda,b,ldb,c,ldc);
            // gemm_4x4_neon(m,n,k,a,lda,b,ldb,c,ldc);
            // gemm_4x4_neon_block(m,n,k,a,lda,b,ldb,c,ldc);
            // gemm_4x4_neon_packed(m,n,k,a,lda,b,ldb,c,ldc);
            // gemm_4x4_like_blas_neon(m,n,k,a,lda,b,ldc,c,ldc);
            // gemm_4x4_like_blas_asmV1(m,n,k,a,lda,b,ldb,c,ldc);
            // gemm_4x4_like_blas_asmV2(m,n,k,a,lda,b,ldb,c,ldc);
            // gemm_4x4_like_blas_asmV3(m,n,k,a,lda,b,ldb,c,ldc);
            gemm_8x8_like_blas_neon(m,n,k,a,lda,b,ldb,c,ldc);
            clock_gettime(CLOCK_MONOTONIC_RAW,&end);
            
            time_used = get_time(&start,&end);
            best_time = std::min(best_time,time_used);
        }
        diff = compare_matrix(m,n,c,ldc,nowc,ldc);
        if(diff > 1e-2){
            std::cout << "diff:[" << diff << "]" <<" program probably has error!"<< std::endl;
            break;
        }
        double GFLOPS = GFLOPs / best_time;
        std::cout << p << " " << GFLOPS << " " << diff << std::endl;
        // std::cout <<"GFLOPs: " << GFLOPs <<" " <<"time_used: " << time_used << std::endl;
        // std::cout << std::fixed << std::setprecision(6) << "GFLOPS:" << GFLOPs*1.0/best_time << std::endl;  
                
        // if(flag){
        //     std::cout<<"A:" <<std::endl;
        //     for(int i=0;i<10;i++){
        //         for(int j=0;j<10;j++){
        //             std::cout <<  *(a + i*ldc +j) << " ";
        //         }
        //         std::cout << std::fixed << std::setprecision(6) << std::endl;
        //     }

        //     std::cout<<"NOWC:" <<std::endl;
        //     for(int i=0;i<10;i++){
        //         for(int j=0;j<10;j++){
        //             std::cout <<  *(nowc + i*ldc +j) << " ";
        //         }
        //         std::cout << std::fixed << std::setprecision(6) << std::endl;
        //     }

        //     std::cout<<"C:" <<std::endl;
        //     for(int i=0;i<10;i++){
        //         for(int j=0;j<10;j++){
        //             std::cout <<  *(c + i*ldc +j) << " ";
        //         }
        //         std::cout << std::fixed << std::setprecision(6) << std::endl;
        //     }
        //     flag=false;
        // }
    }
    delete []c;
    delete []b;
    delete []a;
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <arm_neon.h>
#include <math.h>
#include <sys/time.h>

#define BLOCK_SIZE 4
// static double gtod_ref_time_sec = 0.0;
double dclock()
{
        double the_time, norm_sec;
        struct timeval tv;

        gettimeofday(&tv, NULL);

        // if (gtod_ref_time_sec == 0.0)
        //         gtod_ref_time_sec = (double)tv.tv_sec;

        norm_sec = (double)tv.tv_sec; //- gtod_ref_time_sec;

        the_time = norm_sec + tv.tv_usec * 1.0e-6;

        return the_time;
}


// 普通的矩阵乘
void matrix_multiply_c(float32_t* A,float32_t* B,float32_t* C,uint32_t n,uint32_t m,uint32_t k){
    int i,j,p;
    for(i=0;i<n;i++){
        for(j=0;j<m;j++){
            C[i+j*n] = 0;
            // Inner Product
            for(p=0;p<k;p++){
                C[i+j*n] += A[i + p*n] * B[p + j*k]; 
            }
        }
    }
}

// 用NEON加速的矩阵乘
void matrix_multiply_neon(float32_t *A,float32_t *B,float32_t *C,uint32_t n,uint32_t m,uint32_t k){
    // A,B,C三个阵的逐列数据
    float32x4_t A0,A1,A2,A3;
    float32x4_t B0,B1,B2,B3;
    float32x4_t C0,C1,C2,C3;

    // A,B,C三个阵的地址偏移值
    int a_offset,b_offset,c_offset;
    
    // 迭代变量
    int i_idx,j_idx,k_idx;
    
    for(i_idx=0;i_idx<n;i_idx+=4){
        for(j_idx=0;j_idx<m;j_idx+=4){
            C0 = vmovq_n_f32(0);
            C1 = vmovq_n_f32(0);
            C2 = vmovq_n_f32(0);
            C3 = vmovq_n_f32(0);
            // 得等这一个循环完成了才算做完4x4次内积,因此里面的C的值得累加,出循环再store
            for(k_idx=0;k_idx<k;k_idx+=4){
                // 计算偏移值
                a_offset = i_idx + n*k_idx;
                b_offset = k_idx + k*j_idx;
                // 获取当前基址+偏移值的位置处的列值
                A0 = vld1q_f32(A+a_offset);
                A1 = vld1q_f32(A+a_offset+n);
                A2 = vld1q_f32(A+a_offset+2*n);
                A3 = vld1q_f32(A+a_offset+3*n);

                B0 = vld1q_f32(B+b_offset);
                C0 = vfmaq_laneq_f32(C0,A0,B0,0);
                C0 = vfmaq_laneq_f32(C0,A1,B0,1);
                C0 = vfmaq_laneq_f32(C0,A2,B0,2);
                C0 = vfmaq_laneq_f32(C0,A3,B0,3);

                B1 = vld1q_f32(B+b_offset+k);
                C1 = vfmaq_laneq_f32(C1,A0,B1,0);
                C1 = vfmaq_laneq_f32(C1,A1,B1,1);
                C1 = vfmaq_laneq_f32(C1,A2,B1,2);
                C1 = vfmaq_laneq_f32(C1,A3,B1,3);                                

                B2 = vld1q_f32(B+b_offset+2*k);
                C2 = vfmaq_laneq_f32(C2,A0,B2,0);
                C2 = vfmaq_laneq_f32(C2,A1,B2,1);
                C2 = vfmaq_laneq_f32(C2,A2,B2,2);
                C2 = vfmaq_laneq_f32(C2,A3,B2,3);

                B3 = vld1q_f32(B+b_offset+3*k);
                C3 = vfmaq_laneq_f32(C3,A0,B3,0);
                C3 = vfmaq_laneq_f32(C3,A1,B3,1);
                C3 = vfmaq_laneq_f32(C3,A2,B3,2);
                C3 = vfmaq_laneq_f32(C3,A3,B3,3);
            }
            // 计算偏移值
            c_offset = i_idx + n*j_idx;
            vst1q_f32(C+c_offset,C0);
            vst1q_f32(C+c_offset+n,C1);
            vst1q_f32(C+c_offset+n*2,C2);
            vst1q_f32(C+c_offset+n*3,C3);
        }
    }
}

void print_matrix(float32_t* M,uint32_t row,uint32_t col){
    int i,j;
    printf("===MATRIX DISPLAY START===\n");
    for(i=0;i<row;i++){
        for(j=0;j<col;j++){
            printf("%.6f ",M[i+j*row]);
        }
        printf("\n");
    }
    printf("===MATRIX DISPLAY FINISH===\n");
}

void init_matrix_rand(float32_t* M,uint32_t nums){
    int i;
    for(i=0;i<nums;i++)
        M[i] = (float)rand() / (float)(RAND_MAX);
}

void init_matrix_zero(float32_t* M,uint32_t nums){
    int i;
    for(i=0;i<nums;i++)
        M[i] = 0;
}

bool is_ele_equal(float32_t a,float32_t b){
    if(fabs(a-b) < 1e-6)
        return true;
    return false;
}

bool compare_matrix(float32_t *A,float32_t *B,uint32_t row,uint32_t col){
    int i,j,a,b;
    for(i=0;i<row;i++){
        for(j=0;j<col;j++){
            a = A[i+j*row];
            b = B[i+j*row];
            if(!is_ele_equal(a,b))
                return false;
        }
    }
    return true;
}

int main(){
    // 设置n,m,k的维度,出于方便,都设置为4的倍数
    uint32_t n = 256 * BLOCK_SIZE;
    uint32_t m = 256 * BLOCK_SIZE;
    uint32_t k = 256 * BLOCK_SIZE;

    // 设置A,B,C,D矩阵
    float32_t A[n*k];
    float32_t B[k*m];
    float32_t C[n*m];
    float32_t D[n*m];
    // 计时
    double tc,td,t_start,t_end;

    // 初始化A,B阵,并打印
    init_matrix_rand(A,n*k);
    // print_matrix(A,n,k);
    
    init_matrix_rand(B,k*m);
    // print_matrix(B,k,m);

    // 计算D阵
    t_start = dclock();
    matrix_multiply_c(A,B,D,n,m,k);
    t_end = dclock();
    td = t_end - t_start;
    // print_matrix(D,n,m);

    // 计算C阵
    t_start = dclock();
    matrix_multiply_neon(A,B,C,n,m,k);
    t_end = dclock();
    tc = t_end - t_start;
    // print_matrix(C,n,m);

    // 输出是否相同
    char* is_equal = compare_matrix(A,B,n,m)?"相同":"不相同";
    printf("===C,D阵计算结果%s===\n",is_equal);
    printf("===tc:%.5lf td:%.5lf===\n",tc,td);
    return 0;
}
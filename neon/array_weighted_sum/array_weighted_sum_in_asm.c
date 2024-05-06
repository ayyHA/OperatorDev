#include <stdio.h>
#include <arm_neon.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string>
using namespace std;


void arrayWeightSumInAsm(float* array1,float weighted1,
                         float* array2,float weighted2,
                         int len,float* dst){
    int neon_len = len >> 2;
    int remain_len = len - (neon_len << 2);
    printf("neon_len:%d,remain_len:%d\n",neon_len,remain_len);
#ifdef __aarch64__ // armv8
    __asm__ volatile(
            //  Instruction
        // 取64位寄存器的低32位,并dup到向量寄存器上
        "mov x0, %[weighted1] \n"
        "dup v0.4s, w0 \n"
        "mov x1, %[weighted2] \n"
        "dup v1.4s, w1 \n"
        "0:            \n"
        // 预取数据,加载数据
        "prfm pldl1keep, [%[array1], #128] \n"
        "ld1 {v2.4s}, [%[array1]], #16 \n"
        "prfm pldl1keep, [%[array2], #128] \n"
        "ld1 {v3.4s}, [%[array2]], #16 \n"
        // 做乘法
        "fmul v4.4s, v0.4s, v2.4s \n"
        "fmul v5.4s, v1.4s, v3.4s \n"
        // 做加法
        "fadd v4.4s, v4.4s, v5.4s \n"
        // 存数据
        "st1 {v4.4s}, [%[dst]], #16 \n"
        // 自减neon_len,并影响状态位
        "subs %[neon_len], %[neon_len], #1 \n"
        // 满足条件则跳转
        "bgt 0b \n"
        :[array1]    "+r"(array1),
         [array2]    "+r"(array2),
         [dst]       "+r"(dst),
         [neon_len]  "+r"(neon_len)
        :[weighted1] "r"(weighted1),
         [weighted2] "r"(weighted2)
        :"cc","memory","x0","x1","v0","v1","v2","v3","v4","v5"
    );
#else              // armv7
    __asm__ volatile(
    "vdup.f32   q0, %[arr1Weight]        \n"
    "vdup.f32   q1, %[arr2Weight]        \n"

    "0:                                  \n"
    "pld        [%[arr1Ptr], #128]       \n"
    "vld1.f32   {d4-d5}, [%[arr1Ptr]]!   \n"

    "pld        [%[arr2Ptr], #128]       \n"
    "vld1.f32   {d6-d7}, [%[arr2Ptr]]!   \n"
    
    "vmul.f32   q4, q0, q2 \n"
    "vmul.f32   q5, q1, q3 \n"

    "vadd.f32   q6, q4, q5 \n"

    "subs       %[neonLen], #1                \n"
    
    "vst1.f32   {d12-d13}, [%[resultArrPtr]]! \n"
    
    "bgt        0b                  \n"
    :[arr1Ptr]        "=r"(array1),
     [arr2Ptr]        "=r"(array2),
     [resultArrPtr]   "=r"(dst),
     [neonLen]        "=r"(neon_len)
    :[arr1Weight]     "r"(weighted1),
     [arr2Weight]     "r"(weighted2)
    :"cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
    );
#endif
    // 处理尾部数据
    while(remain_len>0){
        *dst = *array1 * weighted1 + *array2 * weighted2;
        dst++;
        array1++;
        array2++;
        remain_len--;
    }
}
void arrayWeightSumNeon(float* array1,float weighted1,
                      float* array2,float weighted2,
                      int len,float* dst){
    int neon_len = len >> 2;    // /4,因为intrinsic一次处理四个float
    int remain_len = len - neon_len << 2;   // 相当于对4取余了
    
    float32x4_t varr1,varr2,vdst;
    float32x4_t vweight1 = vdupq_n_f32(weighted1);
    float32x4_t vweight2 = vdupq_n_f32(weighted2);
    for(int i=0;i<neon_len;i++){
        varr1 = vld1q_f32(array1);
        varr2 = vld1q_f32(array2);

        varr1 = vmulq_f32(varr1,vweight1);
        varr2 = vmulq_f32(varr2,vweight2);

        vdst = vaddq_f32(varr1,varr2);
        vst1q_f32(dst,vdst);

        array1+=4;
        array2+=4;
        dst+=4;
    }

    // 处理余数
    for(;remain_len>0;remain_len--){
        *dst = *array1 * weighted1 + *array2 * weighted2;
        dst++;
        array1++;
        array2++;
    }
}

void arrayWeightSum(float* array1,float weighted1,
                      float* array2,float weighted2,
                      int len,float* dst){
    int i;
    for(i=0;i<len;i++)
        dst[i] = weighted1 * array1[i] + weighted2 * array2[i];
}

void generate_array(float* array,int len){
    for(int i=0;i<len;i++)
        array[i] = (float)rand() / RAND_MAX;
}

void zero_array(float* array,int len){
    for(int i=0;i<len;i++)
        array[i] = 0;
}

void print_array(float* array,int len){
    for(int i=0;i<len;i++)
        printf("%.3f ",array[i]);
    printf("\n");
}

bool compare_array(float* array1, float* array2,int len){
    for(int i=0;i<len;i++){
        if(fabs(array1[i]-array2[i]) > 1e-6)
            return false;
    }
    return true;
}

int main(){
    float array1[10],array2[10],dst1[10],dst2[10],dst3[10];
    generate_array(array1,10);
    generate_array(array2,10);

    print_array(array1,10);
    print_array(array2,10);
    
    zero_array(dst1,10);
    zero_array(dst2,10);
    zero_array(dst3,10);

    arrayWeightSum(array1,0.5,array2,0.4,10,dst2);
    print_array(dst2,10);

    arrayWeightSumNeon(array1,0.5,array2,0.4,10,dst3);
    print_array(dst3,10);

    arrayWeightSumInAsm(array1,0.5,array2,0.4,10,dst1);
    print_array(dst1,10); 

    // char* str; 
    string str = (compare_array(dst2,dst3,10)) ? "SAME":"UNSAME";
    printf("The result [2 3]: %s\n",str.c_str());
    
    // char* _str;
    string _str = (compare_array(dst2,dst1,10)) ? "SAME":"UNSAME";
    printf("The result [1 2]: %s\n",_str.c_str());
    
    return 0;
}
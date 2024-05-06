/*
    这里是较matrix_multiply_c进阶了一下(依旧列主序)
    从维度小且固定的4x4的矩阵开始优化(感觉跟之前那个SSE的差不多啊)
    之所以选择4x4的优化,是因为我们一个向量寄存器是128位,对于float32_t的类型可以用float32x4_t(类似于__m128)放下


    涉及到的三个intrinsic函数:
    vld1q_f32
    vst1q_f32
    vfmaq_lane_f32    -> 这里写错了应该是vfmaq_laneq_f32
    vmovq_n_f32 // 这个应该是设置初值的吧
*/
void matrix_multiply_4x4_neon(float32_t *A,float32_t *B,float32_t *C,uint32_t n,uint32_t m,uint32_t k){
    // 先把A,B,C的列都给用float32x4_t表示
    float32x4_t A0 = vld1q_f32(A);
    float32x4_t A1 = vld1q_f32(A+4);
    float32x4_t A2 = vld1q_f32(A+8);
    float32x4_t A3 = vld1q_f32(A+12);

    float32x4_t B0,B1,B2,B3;

    // 给结果的中间值逐列设置初值0(因为要累加嘛)
    float32x4_t C0 = vmovq_n_f32(0);
    float32x4_t C1 = vmovq_n_f32(0);
    float32x4_t C2 = vmovq_n_f32(0);
    float32x4_t C3 = vmovq_n_f32(0); 

    B0 = vld1q_f32(B);
    C0 = vfmaq_lane_f32(C0,A0,B0,0);
    C0 = vfmaq_lane_f32(C0,A1,B0,1);
    C0 = vfmaq_lane_f32(C0,A2,B0,2);
    C0 = vfmaq_lane_f32(C0,A3,B0,3);
    vst1q_f32(C,C0);

    B1 = vld1q_f32(B+4);
    C1 = vfmaq_lane_f32(C1,A0,B1,0);
    C1 = vfmaq_lane_f32(C1,A1,B1,1);
    C1 = vfmaq_lane_f32(C1,A2,B1,2);
    C1 = vfmaq_lane_f32(C1,A3,B1,3);
    vst1q_f32(C+4,C1);

    B2 = vld1q_f32(B+8);
    C2 = vfmaq_lane_f32(C2,A0,B2,0);
    C2 = vfmaq_lane_f32(C2,A1,B2,1);
    C2 = vfmaq_lane_f32(C2,A2,B2,2);
    C2 = vfmaq_lane_f32(C2,A3,B2,3);
    vst1q_f32(C+8,C2);

    B3 = vld1q_f32(B+12);
    C3 = vfmaq_lane_f32(C3,A0,B3,0);
    C3 = vfmaq_lane_f32(C3,A1,B3,1);
    C3 = vfmaq_lane_f32(C3,A2,B3,2);
    C3 = vfmaq_lane_f32(C3,A3,B3,3);
    vst1q_f32(C+12,C3);

}
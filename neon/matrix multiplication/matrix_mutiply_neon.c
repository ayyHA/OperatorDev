/**
依旧列主序,只是这时候的矩阵变成了通用情况(若两个维度有不是四的倍数的记得padding)
本质上就是把矩阵划分为4x4大小的小矩阵进行计算
*/
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
                C0 = vfmaq_lane_f32(C0,A0,B0,0);
                C0 = vfmaq_lane_f32(C0,A1,B0,1);
                C0 = vfmaq_lane_f32(C0,A2,B0,2);
                C0 = vfmaq_lane_f32(C0,A3,B0,3);

                B1 = vld1q_f32(B+b_offset+k);
                C1 = vfmaq_lane_f32(C1,A0,B1,0);
                C1 = vfmaq_lane_f32(C1,A1,B1,1);
                C1 = vfmaq_lane_f32(C1,A2,B1,2);
                C1 = vfmaq_lane_f32(C1,A3,B1,3);                                

                B2 = vld1q_f32(B+b_offset+2*k);
                C2 = vfmaq_lane_f32(C2,A0,B2,0);
                C2 = vfmaq_lane_f32(C2,A1,B2,1);
                C2 = vfmaq_lane_f32(C2,A2,B2,2);
                C2 = vfmaq_lane_f32(C2,A3,B2,3);

                B3 = vld1q_f32(B+b_offset+3*k);
                C3 = vfmaq_lane_f32(C3,A0,B3,0);
                C3 = vfmaq_lane_f32(C3,A1,B3,1);
                C3 = vfmaq_lane_f32(C3,A2,B3,2);
                C3 = vfmaq_lane_f32(C3,A3,B3,3);
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
/**
这里没有特别的这种,会在最后弄一个全版
#ifdef __ARM_NEON
#else
#endif 
*/

void arrayWeightedSumNeon(float* array1,float weighted1,
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
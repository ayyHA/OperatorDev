#include "box_filter.h"
/**
    7. 盒子滤波
    行主序 在第6个版本的基础上,利用neon技术进行矢量化

    我们整体分为cache的计算和colsum的计算这两部分
    cache的计算是计算[x-radius,x+radius]这部分的大小
        这部分的计算是取行的2*radius的元素相加,然后!只是往右移动1格,然后利用算法规律加减首尾两端的元素,存在数据依赖,不适合做矢量化
√   colsum是计算[y-radius,y+radius]这部分的大小
        因为我们的colsum是对cache的数据累加,是一整个width的操作,它们都在做着重复的累加操作,
        当然后面有加减首尾两端,但这种加减,是一行对另一行的操作,而不是行内的操作.因此具备并行化的条件
*/
void box_filter_neon(float* src, float* dst, int width, int height, 
                     int radius, vector<float> cache){
    int x,y;
    float* cache_ptr = &cache[0];
    // 实现cache的填充,行内累加
    for(y=0; y<height; y++){
        int sum = 0;
        // i [0,radius) add and put sum
        for(int i=0; i<radius; i++){
            sum += src[y*width + i];
        }

        // x [0,radius] add new right value, put cache_ptr
        for(x=0; x<=radius; x++){
            sum += src[y*width + x + radius];
            cache_ptr[y*width + x] = sum;
        }

        // x [radius+1,width-1-radius] sub old left val, add new right val, put cache_ptr
        for(x=radius+1; x<=width-1-radius; x++){
            sum -= src[y*width + x - radius - 1];
            sum += src[y*width + x + radius];
            cache_ptr[y*width + x] = sum;
        }

        // x [width-radius, width) sub old left val, put cache_ptr
        for(x=width-radius; x<width; x++){
            sum -= src[y*width + x - radius -1];
            cache_ptr[y*width + x] = sum;
        }
    }

    vector<float> colsum(width,0);
    float* colsum_ptr = &colsum[0];

    int neon_len = width >> 2;
    int remain_len = width - (neon_len << 2);


    // i [0,radius) add cache_ptr to colsum_ptr, elementwise
    for(int i=0; i<radius; i++){
        float* tmp_cache_ptr = &cache_ptr[i*width];
        float* tmp_colsum_ptr = colsum_ptr;

        for(int j=0; j<neon_len; j++){
            float32x4_t vcache = vld1q_f32(tmp_cache_ptr);
            float32x4_t vcolsum = vld1q_f32(tmp_colsum_ptr);
            float32x4_t vsum = vaddq_f32(vcache,vcolsum);
            vst1q_f32(tmp_colsum_ptr,vsum);

            tmp_cache_ptr +=4;
            tmp_colsum_ptr +=4;
        }
        for(int j=0; j<remain_len; j++){
            *tmp_colsum_ptr += *tmp_cache_ptr;
            tmp_colsum_ptr++;
            tmp_cache_ptr++;
        }
    }

    // y [0,radius] add new bottom val,
    for(y=0; y<=radius; y++){
        float* tmp_cache_bottom_ptr = &cache_ptr[(y+radius)*width];
        float* tmp_colsum_ptr = colsum_ptr;
        float* tmp_dst = &dst[y*width];
        for(int j=0; j<neon_len; j++){
            float32x4_t vcache = vld1q_f32(tmp_cache_bottom_ptr);
            float32x4_t vcolsum  = vld1q_f32(tmp_colsum_ptr);
            float32x4_t vsum = vaddq_f32(vcache,vcolsum);
            vst1q_f32(tmp_colsum_ptr,vsum);
            vst1q_f32(tmp_dst,vsum);
            tmp_cache_bottom_ptr +=4;
            tmp_colsum_ptr +=4;
            tmp_dst +=4;
        }

        for(int j=0; j<remain_len; j++){
            *tmp_colsum_ptr += *tmp_cache_bottom_ptr;
            *tmp_dst = *tmp_colsum_ptr;
            
            tmp_cache_bottom_ptr++;
            tmp_colsum_ptr++;
            tmp_dst++;
        }
    }

    // y [radius+1,height-1-radius] sub old top val, add new bottom val
    for(y=radius+1; y<=height-1-radius; y++){
        float* tmp_cache_top_ptr = &cache_ptr[(y-radius-1)*width]; 
        float* tmp_cache_bottom_ptr = &cache_ptr[(y+radius)*width];
        float* tmp_dst = &dst[y*width];
        float* tmp_colsum_ptr = colsum_ptr;
        
        for(int j=0; j<neon_len; j++){
            float32x4_t vcache_top = vld1q_f32(tmp_cache_top_ptr);
            float32x4_t vcache_bottom = vld1q_f32(tmp_cache_bottom_ptr);
            float32x4_t vcolsum = vld1q_f32(tmp_colsum_ptr);
            float32x4_t vsum = vsubq_f32(vcolsum,vcache_top);
            vsum = vaddq_f32(vsum,vcache_bottom);
            vst1q_f32(tmp_colsum_ptr,vsum);
            vst1q_f32(tmp_dst,vsum);

            tmp_cache_bottom_ptr+=4;
            tmp_cache_top_ptr+=4;
            tmp_colsum_ptr+=4;
            tmp_dst+=4;
        }

        for(int j=0; j<remain_len;j++){
            *tmp_colsum_ptr -= *tmp_cache_top_ptr;
            *tmp_colsum_ptr += *tmp_cache_bottom_ptr;
            *tmp_dst = *tmp_colsum_ptr;

            tmp_cache_bottom_ptr++;
            tmp_cache_top_ptr++;
            tmp_colsum_ptr++;
            tmp_dst++; 
        }
    }

    // y [height-radius,height) sub old top val
    for(y=height-radius; y<height; y++){
        float* tmp_cache_top_ptr = &cache_ptr[(y-radius-1)*width];
        float* tmp_colsum_ptr = colsum_ptr;
        float* tmp_dst = &dst[y*width];
        
        for(int j=0; j<neon_len; j++){
            float32x4_t vcache = vld1q_f32(tmp_cache_top_ptr);
            float32x4_t vcolsum = vld1q_f32(tmp_colsum_ptr);
            float32x4_t vsum = vsubq_f32(vcolsum,vcache);        
            vst1q_f32(tmp_colsum_ptr,vsum);
            vst1q_f32(tmp_dst,vsum);

            tmp_cache_top_ptr+=4;
            tmp_colsum_ptr+=4;
            tmp_dst+=4;
        }

        for(int j=0; j<remain_len; j++){
            *tmp_colsum_ptr -= *tmp_cache_top_ptr;
            *tmp_dst = *tmp_colsum_ptr;

            tmp_cache_top_ptr++;
            tmp_colsum_ptr++;
            tmp_dst++;
        }
    }
}
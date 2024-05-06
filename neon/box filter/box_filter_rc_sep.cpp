#include "box_filter.h"
/**
    2. 盒子滤波
    行主序, 行列分离版
    减少时间开销即减少嵌套的循环层数(O(n^4) -> 2*O(n^3))

    参数说明:
    cache在外边被cache.resize(width*height)这么个大小了
*/
void box_filter_rc_sep(float* src,float* dst,int width,int height,int radius,vector<float> cache){
    int x,y;
    float* cache_ptr = &cache[0];
    // 做一个width*height这么宽高大小的cache区域,并以行累和值填充之
    for(y = 0; y<height; y++){
        for(x = 0; x<width; x++){
            int start_x = x - radius;
            int end_x = x + radius;
            if(start_x < 0)
                start_x = 0;
            if(end_x >= width)
                end_x = width - 1;
            
            int sum = 0;
            for(int tx=start_x; tx<=end_x;tx++){
                sum += src[y*width + tx];
            }
            cache_ptr[y*width + x] = sum;
        }
    }

    // 以缓冲区的值进行列累和
    for(y=0; y<height; y++){
        int start_y = y - radius;
        int end_y = y + radius;
        if(start_y < 0)
            start_y = 0;
        if(end_y >= height)
            end_y = height - 1;

        for(x=0; x<width; x++){
            int sum = 0;
            for(int ty=start_y; ty <= end_y; ty++){
                sum += cache_ptr[ty*width + x];
            }
            dst[y*width + x] = sum;
        }
    }
}
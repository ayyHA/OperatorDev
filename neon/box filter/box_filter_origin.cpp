/**
    1. 盒子滤波
    行主序,基础的C写法
*/
#include "box_filter.h"
void box_filter_origin(float* src,float* dst,int width,int height,int radius){
    int x,y;
    for(y=0; y<height; y++){
        int start_y = y - radius, end_y = y + radius;
        if(start_y < 0)
            start_y = 0;
        if(end_y >= height)
            end_y = height - 1;
        
        for(x=0; x<width; x++){
            int start_x = x - radius, end_x = x + radius;
            if(start_x < 0)
                start_x = 0;
            if(end_x>=width)
                end_x = width - 1;
            
            int sum=0; 
            for(int ty=start_y;ty<=end_y;ty++){
                for(int tx=start_x;tx<=end_x;tx++){
                    sum += src[ty*width + tx];
                }
            }
            dst[y*width + x] = sum;
        }
    }
}
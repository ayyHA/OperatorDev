#include "box_filter.h"
/**
    5. 盒子滤波
    行主序 增加openCV的改进方式
    前面第3版的方法存在大量cache miss,第4版只是在第3版的基础上进行了规避

    本方法则是仿造opencv的思想构建一个colsum来收集[y-radius,y+radius]里面的列和,收集width个
    那么最终只需要累加对应的列和[x-radius,x+radius]即可,而后则是减去旧的左端值,加上新的右端值
    这样就用上了行主序的矩阵的空间局部性
*/
void box_filter_like_cv(float* src, float* dst, int width, int height,
                        int radius, vector<float> cache){
    int x,y;
    vector<float> colsum(width,0);

    // i∈[0,radius)累和值到colsum里面
    for(int i=0; i<radius; i++){
        for(x=0; x<width; x++){
            colsum[x] += src[i*width + x];
        }
    }

    // y∈[0,radius]只需要累加最新行的值到colsum里面
    for(y=0; y<=radius; y++){
        for(x=0; x<width; x++){
            colsum[x] += src[(y+radius)*width + x];
        }

        int sum = 0;
        // i∈[0,radius)累值进入sum里面
        for(int i=0; i<radius; i++){
            sum += colsum[i];
        }    

        // x∈[0,radius]只加右端新值,即可放入dst
        for(x=0; x<=radius; x++){
            sum += colsum[x+radius];
            dst[y*width + x] = sum;
        }

        // x∈[radius+1,width-1-radius]减左端旧值,加右端新值,放入dst
        for(x=radius+1; x<=width-1-radius; x++){
            sum -= colsum[x-radius-1];
            sum += colsum[x+radius];
            dst[y*width + x] = sum;
        }

        // x∈[width-radius,width)减左端旧值,放入dst
        for(x=width-radius; x<width; x++){
            sum -= colsum[x-radius-1];
            dst[y*width + x] = sum;
        }
    }

    // y∈[radius+1,height-1-radius]colsum减去上端旧行值,加上下端新行值
    for(y=radius+1; y<=height-1-radius; y++){
        // 更新colsum的值
        for(x=0; x<width; x++){
            colsum[x] -= src[(y-radius-1)*width + x]; 
            colsum[x] += src[(y+radius)*width + x];
        }

        int sum = 0;
        // i∈[0,radius)累值进入sum里面
        for(int i=0; i<radius; i++){
            sum += colsum[i];
        }    

        // x∈[0,radius]只加右端新值,即可放入dst
        for(x=0; x<=radius; x++){
            sum += colsum[x+radius];
            dst[y*width + x] = sum;
        }

        // x∈[radius+1,width-1-radius]减左端旧值,加右端新值,放入dst
        for(x=radius+1; x<=width-1-radius; x++){
            sum -= colsum[x-radius-1];
            sum += colsum[x+radius];
            dst[y*width + x] = sum;
        }

        // x∈[width-radius,width)减左端旧值,放入dst
        for(x=width-radius; x<width; x++){
            sum -= colsum[x-radius-1];
            dst[y*width + x] = sum;
        }   
    }

    // y∈[height-radius,height)colsum减上上端旧行值
    for(y=height-radius; y<height; y++){
        // 更新colsum的值
        for(x=0; x<width; x++){
            colsum[x] -= src[(y-radius-1)*width + x];
        }

        int sum = 0;
        // i∈[0,radius)累值进入sum里面
        for(int i=0; i<radius; i++){
            sum += colsum[i];
        }    

        // x∈[0,radius]只加右端新值,即可放入dst
        for(x=0; x<=radius; x++){
            sum += colsum[x+radius];
            dst[y*width + x] = sum;
        }

        // x∈[radius+1,width-1-radius]减左端旧值,加右端新值,放入dst
        for(x=radius+1; x<=width-1-radius; x++){
            sum -= colsum[x-radius-1];
            sum += colsum[x+radius];
            dst[y*width + x] = sum;
        }

        // x∈[width-radius,width)减左端旧值,放入dst
        for(x=width-radius; x<width; x++){
            sum -= colsum[x-radius-1];
            dst[y*width + x] = sum;
        }
    }
}

/**
    6. 盒子滤波
    行主序
    本质上和上面的方法差不多
    不过上面的是先累加列的,再把列和累加,冗余代码有点多

    下面的方法是先累加行的,放到cache里去,然后再累加列的
*/
void box_filter_like_cv_2(float* src, float* dst, int width, int height, 
                          int radius, vector<float> cache){
    int x,y;
    float* cache_ptr = &cache[0];

    // 填充cache_ptr
    for(y=0; y<height; y++){
        int sum = 0;
        // i [0,radius) init
        for(int i=0; i<radius; i++){
            sum += src[y*width + i];
        }

        // x [0,radius] add new right value
        for(x=0; x<=radius; x++){
            sum += src[y*width + x + radius];
            cache_ptr[y*width + x] = sum;   
        }

        // x [radius+1,width-1-radius] sub old left value, add new right value
        for(x=radius+1; x<=width-1-radius; x++){
            sum -= src[y*width + x - radius - 1];
            sum += src[y*width + x + radius];
            cache_ptr[y*width + x] = sum;
        }

        // x [width-radius,width) sub old left value
        for(x=width-radius; x<width; x++){
            sum -= src[y*width + x - radius - 1];
            cache_ptr[y*width + x] = sum;
        }
    }

    vector<float> colsum(width,0);
    // 整行的进行累加,利用空间局部性
    // y [0,radius) init colsum
    for(y=0; y<radius; y++){
        for(x=0; x<width; x++){
            colsum[x] += cache_ptr[y*width + x];
        }
    }

    // y [0,radius] add new bottom value, put dst
    for(y=0; y<=radius; y++){
        for(x=0; x<width; x++){
            colsum[x] += cache_ptr[(y+radius)*width + x];
            dst[y*width + x] = colsum[x];
        }
    }

    // y [radius+1,height-1-radius] sub old top value, add new bottom value, put dst
    for(y=radius+1; y<=height-1-radius; y++){
        for(x=0; x<width; x++){
            colsum[x] -= cache_ptr[(y-radius-1)*width + x];
            colsum[x] += cache_ptr[(y+radius)*width + x];
            dst[y*width + x] = colsum[x];
        }
    }

    // y [height-radius, height) sub old top value, put dst
    for(y=height-radius; y<height; y++){
        for(x=0; x<width; x++){
            colsum[x] -= cache_ptr[(y-radius-1)*width + x];
            dst[y*width + x] = colsum[x];
        }
    }
}
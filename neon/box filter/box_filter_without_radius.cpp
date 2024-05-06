#include "box_filter.h"
/**
    3. 盒子滤波
    行主序,在第2版行列分离的基础上,减少冗余计算

    不难发现虽然第二版拆分了行列,减少了循环的层数,但是其实数据是被重复计算的
    以构建行累和值为例,两个相邻的数算的数据分别为:[x-radius,x+radius],[x-radius+1,x+radius+1]
    即对于右移一位的累和值而言:整体的计算是减去左端点-1的值,再加上新的右端点的值,而中间的部分是已有的结果

    通过这样操作,拆分radius的循环计算为至多两次的加减计算,大大减少计算次数,同时由于:
    第三层循环[start_x,end_x]操作被拆分,因此可以将该循环去除,交由x进行,则O(n^3) -> O(n^2)
    
    同时沿高度方向也可以这样做
*/

void box_filter_without_radius(float* src, float* dst, int width, int height, 
                               int radius, vector<float> cache){
    int x,y;
    float* cache_ptr = &cache[0];

    // 填充cache_ptr
    for(y=0; y<height; y++){
        cout << "进入水平填充" << endl;
        int sum = 0;
        // i∈[0,radius)对sum累和
        for(int i=0; i<radius; i++)
            sum += src[y*width + i];
        
        // x∈[0,radius]加上新的右端值累和,并写入cache_ptr
        for(x=0; x<=radius; x++){
            sum += src[y*width + x + radius];
            cache_ptr[y*width + x] = sum;
        }

        // x∈[radius+1,width-1-radius]加上新的右端值累和,减去旧的左端值(即新的左端值-1),并写入cache_ptr
        for(x=radius+1; x<=width-1-radius; x++){
            sum -= src[y*width + x - radius - 1];
            sum += src[y*width + x + radius];
            cache_ptr[y*width + x] = sum;
        }

        // x∈[width-radius,width)减去旧的左端值,并写入cache_ptr
        for(x=width-radius; x<width; x++){
            sum -= src[y*width + x - radius - 1];
            cache_ptr[y*width + x] = sum;
        }
        cout << "离开水平填充" << endl;
    }
    for(x=0; x<width; x++){
        int sum=0;
        cout << "进入垂直填充"<<endl;
        // i∈[0,radius)对sum累和,采用cache_ptr值进行累和
        for(int i=0; i<radius; i++)
            sum += cache_ptr[i*width + x];
    
        // y∈[0,radius]增加新的下端值,并写入dst
        for(y=0; y<=radius; y++){
            sum += cache_ptr[(y+radius)*width + x];
            dst[y*width + x] = sum;   
        }

        // y∈[radius+1,height-1-radius]减去旧的上端值,加上新的下端值,并写入dst
        for(y=radius+1; y<=height-1-radius; y++){
            sum -= cache_ptr[(y-radius-1)*width + x];
            sum += cache_ptr[(y+radius)*width + x];
            dst[y*width + x] = sum;
        }

        // y∈[height-radius,height)减去旧的上端值,写入dst
        for(y=height-radius; y<height; y++){
            sum -= cache_ptr[(y-radius-1)*width + x];
            dst[y*width + x] = sum;
        }
        cout << "离开垂直填充"<<endl;
    }
}

/**
    4. 盒子滤波
    行主序,第3版的第2个二层循环是从x及y的,即x>y(表示x到y),存在如下问题:
    因为y是内层循环的参数,而y的相邻位置变换又要乘上一个width:
    以最坏极端情况考虑,一个width的数据刚好是cache line的大小,则每一次遍历内循环,就要发生一次cache miss(假设已填充满缓冲区)
    那看来这样很不划算
    因此下面的第四种优化是把第三种优化的优化方式和第二种的combine在一起
*/
void box_filter_row_without_radius(float* src, float* dst, int width, int height, 
                                   int radius, vector<float> cache){
    int x,y;
    float* cache_ptr = &cache[0];

    for(y=0; y<height; y++){
        int sum = 0;

        // i∈[0,radius) 初始化sum
        for(int i=0; i<radius; i++)
            sum += src[y*width + i];

        // x∈[0,radius] 只加右端新值
        for(x=0; x<=radius; x++){
            sum += src[y*width + x + radius];
            cache_ptr[y*width + x] = sum;
        }

        // x∈[radius+1,width-1-radius] 减左端旧值,加右端新值
        for(x=radius+1; x<=width-1-radius; x++){
            sum -= src[y*width + x - radius - 1];
            sum += src[y*width + x + radius];
            cache_ptr[y*width + x] = sum;
        }

        // x∈[width-radius,width) 减左端旧值
        for(x=width-radius; x<width; x++){
            sum -= src[y*width + x - radius - 1];
            cache_ptr[y*width + x] = sum;
        }
    }

    // 这里为了避免cache miss的问题换成第2版的方法
    for(y=0; y<height; y++){
        int start_y = y - radius;
        int end_y = y + radius;
        if(start_y < 0)
            start_y = 0;
        if(end_y >= height)
            end_y = height - 1;

        // 利用空间局部性减少cache miss
        for(x=0; x<width; x++){
            int sum = 0;
            for(int ty=start_y; ty<=end_y; ty++){
                sum += cache_ptr[ty*width + x];
            }
            dst[y*width + x] = sum;
        }
    }
}

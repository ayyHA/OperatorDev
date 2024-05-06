#include "box_filter.h"
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main(){
    Mat src = cv::imread("./test2.jpg",0);   
    int height = src.rows;
    int width = src.cols;
    int radius = 3;
    unsigned char *src_tmp = src.data;    
    float* src_ptr = new float[width*height];
    float* dest_ptr = new float[width*height];
    
    for(int i=0;i<width*height;i++)
        src_ptr[i] = (float)(src_tmp[i]);
    
    for(int x=0;x<20;x++){
        for(int y=0;y<20;y++){
            cout << src_ptr[x*width+y] << " ";
        }
        cout << endl;
    }

    vector<float> cache(width*height,0);
    long long st = cv::getTickCount();
    // box_filter_origin(src_ptr,dest_ptr,width,height,radius);                     // test1.png: 836.676ms // test2.jpg: 4940.26ms
    // box_filter_rc_sep(src_ptr,dest_ptr,width,height,radius,cache);               // test1.png: 276.814ms // test2.jpg: 1624.86ms
    // box_filter_without_radius(src_ptr,dest_ptr,width,height,radius,cache);       // test1.png: 320.094ms // test2.jpg: 1503.69ms
    // box_filter_row_without_radius(src_ptr,dest_ptr,width,height,radius,cache);   // test1.png: 182.202ms // test2.jpg: 1087.14ms
    // box_filter_like_cv(src_ptr,dest_ptr,width,height,radius,cache);              // test1.png: 74.1177ms // test2.jpg: 437.618ms
    // box_filter_like_cv_2(src_ptr,dest_ptr,width,height,radius,cache);            // test1.png: 88.5152ms // test2.jpg: 492.579ms
    // box_filter_neon(src_ptr,dest_ptr,width,height,radius,cache);                 // test1.png: 62.3104ms // test2.jpg: 355.961ms
    // box_filter_in_asm(src_ptr,dest_ptr,width,height,radius,cache);               // test1.png: 49.5036ms // test2.jpg: 286.974ms    
    box_filter_in_asmv2(src_ptr,dest_ptr,width,height,radius,cache);                // test1.png: 46.1996ms // test2.jpg: 275.214ms
    double duration = (cv::getTickCount() - st) / cv::getTickFrequency() * 1000;
    cout << "width: " << width << " height: " << height <<endl;
    cout << "duration: " << duration << "ms" <<endl;

    for(int x=0;x<20;x++){
        for(int y=0;y<20;y++){
            cout << dest_ptr[x*width+y] << " ";
        }
        cout << endl;
    }

    // vector<float> cache(width*height,0);
    // box_filter_neon(src_ptr,dest_ptr,width,height,radius,cache);

    delete src_ptr;
    delete dest_ptr;

    return 0;
}
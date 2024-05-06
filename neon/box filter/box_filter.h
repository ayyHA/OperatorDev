#include <iostream>
#ifdef __ARM_NEON
    #include <arm_neon.h>
#endif
#include<sys/time.h>
#include<vector>
using namespace std;

void box_filter_origin(float* src,float* dst,int width,int height,int radius);
void box_filter_rc_sep(float* src,float* dst,int width,int height,int radius,vector<float> cache);
void box_filter_without_radius(float* src, float* dst, int width, int height, int radius, vector<float> cache);
void box_filter_row_without_radius(float* src, float* dst, int width, int height, int radius, vector<float> cache);
void box_filter_like_cv(float* src, float* dst, int width, int height,int radius, vector<float> cache);
void box_filter_like_cv_2(float* src, float* dst, int width, int height, int radius, vector<float> cache);
void box_filter_neon(float* src, float* dst, int width, int height, int radius, vector<float> cache);
void box_filter_in_asm(float* src,float* dst,int width,int height,int radius,vector<float> cache);
void box_filter_in_asmv2(float* src,float* dst,int width,int height,int radius,vector<float> cache);
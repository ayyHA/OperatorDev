#include "box_filter.h"

void box_filter_in_asm(float* src,float* dst,int width,int height,
                       int radius,vector<float> cache){
    int x,y;
    float *cache_ptr = &cache[0];
    
    // init cache_ptr via horizontal radius add
    for(y=0; y<height; y++){
        int sum = 0;
        // init
        for(int i=0;i<radius;i++){
            sum += src[y*width + i];
        }

        // left
        for(x=0; x<=radius; x++){
            sum += src[y*width + x + radius];
            cache_ptr[y*width + x] = sum;
        }

        // medium
        for(x=radius+1; x<=width-1-radius; x++){
            sum -= src[y*width + x - radius - 1];
            sum += src[y*width + x + radius];
            cache_ptr[y*width + x] = sum;
        }

        // right
        for(x=width-radius; x<width; x++){
            sum -= src[y*width + x - radius - 1];
            cache_ptr[y*width + x] = sum;
        }
    }

    vector<float> colsum(width,0);
    float* colsum_ptr = &colsum[0];

    // vertical radius add
    int neon_len = width >> 2;
    int remain_len = width - (neon_len << 2);

    for(int i=0;i<radius; i++){
        float *tmp_cache_ptr = &cache_ptr[i*width];
        float *tmp_colsum_ptr = colsum_ptr;

        int nl = neon_len;
        int rl = remain_len;

        __asm__ volatile(
            "0:                         \n"
            "ld1 {v0.4s},[%[tmp_colsum_ptr]]    \n"
            "ld1 {v1.4s},[%[tmp_cache_ptr]],#16 \n"
            "fadd v2.4s,v0.4s,v1.4s     \n"
            "st1 {v2.4s},[%[tmp_colsum_ptr]],#16 \n"
            "subs %[neon_len],%[neon_len],#1 \n"
            "bne 0b \n"
            // "1:                         \n"
            // "ldr w0,[%[tmp_colsum_ptr]]   \n"
            // "ldr w1,[%[tmp_cache_ptr]],#4     \n"
            // "add w2,w0,w1              \n"
            // "str w2,[%[tmp_colsum_ptr]],#4  \n"
            // "subs %[remain_len],%[remain_len],#1 \n"
            // "bne 1b \n"
            :[tmp_colsum_ptr]   "+r"(tmp_colsum_ptr),
             [tmp_cache_ptr]    "+r"(tmp_cache_ptr),
             [neon_len]     "+r"(nl)
            //  [remain_len]   "+r"(rl)
            : 
            :"cc","memory","v0","v1","v2"
        );
        for(;rl>0;rl--){
            *tmp_colsum_ptr += *tmp_cache_ptr;
            tmp_colsum_ptr++;
            tmp_cache_ptr++;
        }
    }
    cout <<"aabb"<<endl;
    
    for(y=0; y<=radius; y++){
        float* tmp_cache_bottom_ptr = &cache_ptr[(y+radius)*width];
        float* tmp_colsum_ptr = colsum_ptr;
        float* tmp_dst = &dst[y*width];

        int nl = neon_len;
        int rl = remain_len;

        __asm__ volatile(
            "0:             \n"
            "ld1 {v0.4s},[%[tmp_colsum_ptr]] \n"
            "ld1 {v1.4s},[%[tmp_cache_bottom_ptr]],#16   \n"
            "fadd v2.4s,v0.4s,v1.4s     \n"
            "st1 {v2.4s},[%[tmp_colsum_ptr]],#16    \n"
            "st1 {v2.4s},[%[tmp_dst]],#16 \n"
            "subs %[neon_len],%[neon_len],#1    \n"
            "bne 0b         \n"
            // "1:             \n"
            // "ldr w0,[%[tmp_colsum_ptr]]   \n"
            // "ldr w1,[%[tmp_cache_bottom_ptr]],#4 \n"
            // "add w2,w0,w1  \n"
            // "str w2,[%[tmp_colsum_ptr]],#4  \n"
            // "str w2,[%[tmp_dst]],#4 \n"
            // "subs %[remain_len],%[remain_len],#1    \n"
            // "bne 1b"
            :[tmp_colsum_ptr] "+r"(tmp_colsum_ptr),
             [tmp_cache_bottom_ptr] "+r"(tmp_cache_bottom_ptr),
             [tmp_dst]  "+r"(tmp_dst),
             [neon_len] "+r"(nl)
            //  [remain_len]   "+r"(rl)
            :
            :"cc","memory","v0","v1","v2"
        );
        for(;rl>0;rl--){
            *tmp_colsum_ptr += *tmp_cache_bottom_ptr;
            *tmp_dst = *tmp_colsum_ptr;
            
            tmp_colsum_ptr++;
            tmp_cache_bottom_ptr++;
            tmp_dst++;
        }
    }
    cout << "bbcc"<<endl;

    for(y=radius+1; y<=height-1-radius; y++){
        float *tmp_cache_bottom_ptr = &cache_ptr[(y+radius)*width];
        float *tmp_cache_top_ptr = &cache_ptr[(y-radius-1)*width];
        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_dst = &dst[y*width];

        int nl = neon_len;   
        int rl = remain_len;

        __asm__ volatile(
            "0: \n"
            "ld1 {v0.4s},[%[tmp_colsum_ptr]] \n"
            "ld1 {v1.4s},[%[tmp_cache_bottom_ptr]],#16 \n"
            "ld1 {v2.4s},[%[tmp_cache_top_ptr]],#16 \n"
            "fadd v0.4s,v0.4s,v1.4s \n"
            "fsub v0.4s,v0.4s,v2.4s \n"
            "st1 {v0.4s},[%[tmp_colsum_ptr]],#16 \n"
            "st1 {v0.4s},[%[tmp_dst]],#16 \n"
            "subs %[neon_len],%[neon_len],#1    \n"
            "bne 0b \n"
            // "1: \n"
            // "ldr w0,[%[tmp_colsum_ptr]] \n"
            // "ldr w1,[%[tmp_cache_bottom_ptr]],#4    \n"
            // "ldr w2,[%[tmp_cache_top_ptr]],#4   \n"
            // "add w0,w0,w1  \n"
            // "sub w0,w0,w2  \n"
            // "str w0,[%[tmp_colsum_ptr]],#4  \n"
            // "str w0,[%[tmp_dst]],#4 \n"
            // "subs %[remain_len],%[remain_len],#1    \n"
            // "bne 1b \n"
            :[tmp_cache_bottom_ptr] "+r"(tmp_cache_bottom_ptr),
             [tmp_cache_top_ptr]    "+r"(tmp_cache_top_ptr),
             [tmp_colsum_ptr]       "+r"(tmp_colsum_ptr),
             [tmp_dst]              "+r"(tmp_dst),
             [neon_len]             "+r"(nl)
            //  [remain_len]           "+r"(rl)
            :
            :"cc","memory","v0","v1","v2"
        );
        for(;rl>0;rl--){
            *tmp_colsum_ptr -= *tmp_cache_top_ptr;
            *tmp_colsum_ptr += *tmp_cache_bottom_ptr;
            *tmp_dst = *tmp_colsum_ptr;

            tmp_cache_bottom_ptr++;
            tmp_cache_top_ptr++;
            tmp_colsum_ptr++;
            tmp_dst++; 
        }
    }
    cout << "ccdd" << endl;

    for(y=height-radius; y<height; y++){
        float *tmp_cache_top_ptr = &cache[(y-radius-1)*width];
        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_dst = &dst[y*width];

        int nl = neon_len;
        int rl = remain_len;

        __asm__ volatile(
            "0: \n"
            "ld1 {v0.4s},[%[tmp_colsum_ptr]]    \n"
            "ld1 {v1.4s},[%[tmp_cache_top_ptr]],#16 \n"
            "fsub v0.4s,v0.4s,v1.4s \n"
            "st1 {v0.4s},[%[tmp_colsum_ptr]],#16    \n"
            "st1 {v0.4s},[%[tmp_dst]],#16   \n"
            "subs %[neon_len],%[neon_len],#1    \n"
            "bne 0b \n"
            // "1: \n"
            // "ldr w0,[%[tmp_colsum_ptr]] \n"
            // "ldr w1,[%[tmp_cache_top_ptr]],#4   \n"
            // "sub w0,w0,w1  \n"
            // "str w0,[%[tmp_colsum_ptr]],#4  \n"
            // "str w1,[%[tmp_dst]],#4 \n"
            // "subs %[remain_len],%[remain_len],#1    \n"
            // "bne 1b \n"
            :[tmp_colsum_ptr] "+r"(tmp_colsum_ptr),
             [tmp_cache_top_ptr] "+r"(tmp_cache_top_ptr),
             [tmp_dst] "+r"(tmp_dst),
             [neon_len] "+r"(nl)
            //  [remain_len]   "+r"(rl)
            :
            :"cc","memory","v0","v1"
        );

        for(;rl>0;rl--){
            *tmp_colsum_ptr -= *tmp_cache_top_ptr;
            *tmp_dst = *tmp_colsum_ptr;

            tmp_cache_top_ptr++;
            tmp_colsum_ptr++;
            tmp_dst++;
        }
    }
    cout << "ddee"<<endl;

}



void box_filter_in_asmv2(float* src,float* dst,int width,int height,
                       int radius,vector<float> cache){
    int x,y;
    float *cache_ptr = &cache[0];
    
    // init cache_ptr via horizontal radius add
    for(y=0; y<height; y++){
        int sum = 0;
        // init
        for(int i=0;i<radius;i++){
            sum += src[y*width + i];
        }

        // left
        for(x=0; x<=radius; x++){
            sum += src[y*width + x + radius];
            cache_ptr[y*width + x] = sum;
        }

        // medium
        for(x=radius+1; x<=width-1-radius; x++){
            sum -= src[y*width + x - radius - 1];
            sum += src[y*width + x + radius];
            cache_ptr[y*width + x] = sum;
        }

        // right
        for(x=width-radius; x<width; x++){
            sum -= src[y*width + x - radius - 1];
            cache_ptr[y*width + x] = sum;
        }
    }

    vector<float> colsum(width,0);
    float* colsum_ptr = &colsum[0];

    // vertical radius add
    int neon_len = width >> 2;
    int remain_len = width - (neon_len << 2);

    for(int i=0;i<radius; i++){
        float *tmp_cache_ptr = &cache_ptr[i*width];
        float *tmp_colsum_ptr = colsum_ptr;

        int nl = neon_len;
        int rl = remain_len;

        __asm__ volatile(
            "0:                         \n"
            "prfm pldl1keep,[%[tmp_colsum_ptr],#256]    \n"
            "ld1 {v0.4s},[%[tmp_colsum_ptr]]    \n"
            "prfm pldl1keep,[%[tmp_cache_ptr],#256]     \n"
            "ld1 {v1.4s},[%[tmp_cache_ptr]],#16 \n"
            "fadd v2.4s,v0.4s,v1.4s     \n"
            "st1 {v2.4s},[%[tmp_colsum_ptr]],#16 \n"
            "subs %[neon_len],%[neon_len],#1 \n"
            "bne 0b \n"
            // "1:                         \n"
            // "ldr w0,[%[tmp_colsum_ptr]]   \n"
            // "ldr w1,[%[tmp_cache_ptr]],#4     \n"
            // "add w2,w0,w1              \n"
            // "str w2,[%[tmp_colsum_ptr]],#4  \n"
            // "subs %[remain_len],%[remain_len],#1 \n"
            // "bne 1b \n"
            :[tmp_colsum_ptr]   "+r"(tmp_colsum_ptr),
             [tmp_cache_ptr]    "+r"(tmp_cache_ptr),
             [neon_len]     "+r"(nl)
            //  [remain_len]   "+r"(rl)
            : 
            :"cc","memory","v0","v1","v2"
        );
        for(;rl>0;rl--){
            *tmp_colsum_ptr += *tmp_cache_ptr;
            tmp_colsum_ptr++;
            tmp_cache_ptr++;
        }
    }
    cout <<"aabb"<<endl;
    
    for(y=0; y<=radius; y++){
        float* tmp_cache_bottom_ptr = &cache_ptr[(y+radius)*width];
        float* tmp_colsum_ptr = colsum_ptr;
        float* tmp_dst = &dst[y*width];

        int nl = neon_len;
        int rl = remain_len;

        __asm__ volatile(
            "0:             \n"
            "prfm pldl1keep,[%[tmp_colsum_ptr],#256]    \n"
            "ld1 {v0.4s},[%[tmp_colsum_ptr]] \n"
            "prfm pldl1keep,[%[tmp_cache_bottom_ptr],#256]  \n"
            "ld1 {v1.4s},[%[tmp_cache_bottom_ptr]],#16   \n"
            "fadd v2.4s,v0.4s,v1.4s     \n"
            "st1 {v2.4s},[%[tmp_colsum_ptr]],#16    \n"
            "st1 {v2.4s},[%[tmp_dst]],#16 \n"
            "subs %[neon_len],%[neon_len],#1    \n"
            "bne 0b         \n"
            // "1:             \n"
            // "ldr w0,[%[tmp_colsum_ptr]]   \n"
            // "ldr w1,[%[tmp_cache_bottom_ptr]],#4 \n"
            // "add w2,w0,w1  \n"
            // "str w2,[%[tmp_colsum_ptr]],#4  \n"
            // "str w2,[%[tmp_dst]],#4 \n"
            // "subs %[remain_len],%[remain_len],#1    \n"
            // "bne 1b"
            :[tmp_colsum_ptr] "+r"(tmp_colsum_ptr),
             [tmp_cache_bottom_ptr] "+r"(tmp_cache_bottom_ptr),
             [tmp_dst]  "+r"(tmp_dst),
             [neon_len] "+r"(nl)
            //  [remain_len]   "+r"(rl)
            :
            :"cc","memory","v0","v1","v2"
        );
        for(;rl>0;rl--){
            *tmp_colsum_ptr += *tmp_cache_bottom_ptr;
            *tmp_dst = *tmp_colsum_ptr;
            
            tmp_colsum_ptr++;
            tmp_cache_bottom_ptr++;
            tmp_dst++;
        }
    }
    cout << "bbcc"<<endl;

    for(y=radius+1; y<=height-1-radius; y++){
        float *tmp_cache_bottom_ptr = &cache_ptr[(y+radius)*width];
        float *tmp_cache_top_ptr = &cache_ptr[(y-radius-1)*width];
        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_dst = &dst[y*width];

        int nl = neon_len;   
        int rl = remain_len;

        __asm__ volatile(
            "0: \n"
            "prfm pldl1keep,[%[tmp_colsum_ptr],#256]    \n"
            "ld1 {v0.4s},[%[tmp_colsum_ptr]] \n"
            "prfm pldl1keep,[%[tmp_cache_bottom_ptr],#256]  \n"
            "ld1 {v1.4s},[%[tmp_cache_bottom_ptr]],#16 \n"
            "prfm pldl1keep,[%[tmp_cache_top_ptr],#256] \n"
            "ld1 {v2.4s},[%[tmp_cache_top_ptr]],#16 \n"
            "fadd v0.4s,v0.4s,v1.4s \n"
            "fsub v0.4s,v0.4s,v2.4s \n"
            "st1 {v0.4s},[%[tmp_colsum_ptr]],#16 \n"
            "st1 {v0.4s},[%[tmp_dst]],#16 \n"
            "subs %[neon_len],%[neon_len],#1    \n"
            "bne 0b \n"
            // "1: \n"
            // "ldr w0,[%[tmp_colsum_ptr]] \n"
            // "ldr w1,[%[tmp_cache_bottom_ptr]],#4    \n"
            // "ldr w2,[%[tmp_cache_top_ptr]],#4   \n"
            // "add w0,w0,w1  \n"
            // "sub w0,w0,w2  \n"
            // "str w0,[%[tmp_colsum_ptr]],#4  \n"
            // "str w0,[%[tmp_dst]],#4 \n"
            // "subs %[remain_len],%[remain_len],#1    \n"
            // "bne 1b \n"
            :[tmp_cache_bottom_ptr] "+r"(tmp_cache_bottom_ptr),
             [tmp_cache_top_ptr]    "+r"(tmp_cache_top_ptr),
             [tmp_colsum_ptr]       "+r"(tmp_colsum_ptr),
             [tmp_dst]              "+r"(tmp_dst),
             [neon_len]             "+r"(nl)
            //  [remain_len]           "+r"(rl)
            :
            :"cc","memory","v0","v1","v2"
        );
        for(;rl>0;rl--){
            *tmp_colsum_ptr -= *tmp_cache_top_ptr;
            *tmp_colsum_ptr += *tmp_cache_bottom_ptr;
            *tmp_dst = *tmp_colsum_ptr;

            tmp_cache_bottom_ptr++;
            tmp_cache_top_ptr++;
            tmp_colsum_ptr++;
            tmp_dst++; 
        }
    }
    cout << "ccdd" << endl;

    for(y=height-radius; y<height; y++){
        float *tmp_cache_top_ptr = &cache[(y-radius-1)*width];
        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_dst = &dst[y*width];

        int nl = neon_len;
        int rl = remain_len;

        __asm__ volatile(
            "0: \n"
            "prfm pldl1keep,[%[tmp_colsum_ptr],#256]    \n"
            "ld1 {v0.4s},[%[tmp_colsum_ptr]]    \n"
            "prfm pldl1keep,[%[tmp_cache_top_ptr],#256] \n"
            "ld1 {v1.4s},[%[tmp_cache_top_ptr]],#16 \n"
            "fsub v0.4s,v0.4s,v1.4s \n"
            "st1 {v0.4s},[%[tmp_colsum_ptr]],#16    \n"
            "st1 {v0.4s},[%[tmp_dst]],#16   \n"
            "subs %[neon_len],%[neon_len],#1    \n"
            "bne 0b \n"
            // "1: \n"
            // "ldr w0,[%[tmp_colsum_ptr]] \n"
            // "ldr w1,[%[tmp_cache_top_ptr]],#4   \n"
            // "sub w0,w0,w1  \n"
            // "str w0,[%[tmp_colsum_ptr]],#4  \n"
            // "str w1,[%[tmp_dst]],#4 \n"
            // "subs %[remain_len],%[remain_len],#1    \n"
            // "bne 1b \n"
            :[tmp_colsum_ptr] "+r"(tmp_colsum_ptr),
             [tmp_cache_top_ptr] "+r"(tmp_cache_top_ptr),
             [tmp_dst] "+r"(tmp_dst),
             [neon_len] "+r"(nl)
            //  [remain_len]   "+r"(rl)
            :
            :"cc","memory","v0","v1"
        );

        for(;rl>0;rl--){
            *tmp_colsum_ptr -= *tmp_cache_top_ptr;
            *tmp_dst = *tmp_colsum_ptr;

            tmp_cache_top_ptr++;
            tmp_colsum_ptr++;
            tmp_dst++;
        }
    }
    cout << "ddee"<<endl;

}
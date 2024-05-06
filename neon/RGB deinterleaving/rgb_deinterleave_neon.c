#include <stdio.h>
#include "arm_neon.h"


void rgb_deinterleave_neon(uint8_t* r,uint8_t* g,uint8_t* b,
							uint8_t* rgb,int color_length){
	int num8x16 = color_length/16;
	uint8x16x3_t intlv_rgb;
	for(int i=0;i<num8x16;i++){
		intlv_rgb = vld3q_u8(rgb+3*16*i);
		vst1q_u8(r+16*i,intlv_rgb.val[0]);
		vst1q_u8(g+16*i,intlv_rgb.val[1]);
		vst1q_u8(b+16*i,intlv_rgb.val[2]);
	}
}

int main(){
	return 0;
}

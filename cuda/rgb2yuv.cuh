#ifndef RGB2YUV_CUH
#define RGB2YUV_CUH

#include <stdio.h>
#include <stdint.h>



extern "C" 
void rgb2yuv(const char *src,uint8_t *dest,int width,int height);


#endif
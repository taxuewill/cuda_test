#ifndef RGB2YUV_CUH
#define RGB2YUV_CUH

#include <stdio.h>
#include <stdint.h>



extern "C" 
void rgb2yuv(const char *src,uint8_t *dest,int width,int height);

extern "C"
void rgb2NV12(const char *src,uint8_t *dest,int width,int height);

extern "C"
void cudaNV12ToRGBA(char* srcDev, size_t srcPitch, char* destDev, size_t destPitch, size_t width, size_t height);

#endif
#include "rgb2yuv.cuh"
#include <stdint.h>
#include <cuda_runtime.h>

#define CHECK(res) { if(res != cudaSuccess){printf("Error ：%s:%d , ", __FILE__,__LINE__);   \
printf("code : %d , reason : %s \n", res,cudaGetErrorString(res));exit(-1);}}

#define CUDA(x)				cudaCheckError((x), #x, __FILE__, __LINE__)
#define CUDA_FAILED(x)			(CUDA(x) != cudaSuccess)

#define COLOR_COMPONENT_MASK            0x3FF
#define COLOR_COMPONENT_BIT_SIZE        10

#define FIXED_DECIMAL_POINT             24
#define FIXED_POINT_MULTIPLIER          1.0f
#define FIXED_COLOR_COMPONENT_MASK      0xffffffff

#define LOG_CUDA "[cuda]   "

__constant__ uint32_t constAlpha;
__constant__ float  constHueColorSpaceMat[9];

/**
 * iDivUp
 */
 inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }

 inline cudaError_t cudaCheckError(cudaError_t retval, const char* txt, const char* file, int line )
 {

 
	 //int activeDevice = -1;
	 //cudaGetDevice(&activeDevice);
 
	 //Log("[cuda]   device %i  -  %s\n", activeDevice, txt);
	 
	 printf(LOG_CUDA "%s\n", txt);
 
 
	 if( retval != cudaSuccess )
	 {
		 printf(LOG_CUDA "   %s (error %u) (hex 0x%02X)\n", cudaGetErrorString(retval), retval, retval);
		 printf(LOG_CUDA "   %s:%i\n", file, line);	
	 }
 
	 return retval;
 }
 

inline __device__ void rgb_to_y(const uint8_t r, const uint8_t g, const uint8_t b, uint8_t& y)
{
	y = static_cast<uint8_t>(((int)(30 * r) + (int)(59 * g) + (int)(11 * b)) / 100);
}

inline __device__ void rgb_to_yuv(const uint8_t r, const uint8_t g, const uint8_t b, uint8_t& y, uint8_t& u, uint8_t& v)
{
	rgb_to_y(r, g, b, y);
	u = static_cast<uint8_t>(((int)(-17 * r) - (int)(33 * g) + (int)(50 * b) + 12800) / 100);
	v = static_cast<uint8_t>(((int)(50 * r) - (int)(42 * g) - (int)(8 * b) + 12800) / 100);
}

template <typename T, bool formatYV12>
__global__ void RGB_to_YV12( T* src, int srcAlignedWidth, uint8_t* dst, int dstPitch, int width, int height )
{
	const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    printf("[%d,%d]\n",x,y);
	const int x1 = x + 1;
	const int y1 = y + 1;

	if( x1 >= width || y1 >= height )
		return;

	const int planeSize = height * dstPitch;
	
	uint8_t* y_plane = dst;
	uint8_t* u_plane;
	uint8_t* v_plane;

	if( formatYV12 )
	{
		u_plane = y_plane + planeSize;
		v_plane = u_plane + (planeSize / 4);	// size of U & V planes is 25% of Y plane
	}
	else
	{
		v_plane = y_plane + planeSize;		// in I420, order of U & V planes is reversed
		u_plane = v_plane + (planeSize / 4);
	}

	T px;
	uint8_t y_val, u_val, v_val;

	px = src[y * srcAlignedWidth + x];
	rgb_to_y(px.x, px.y, px.z, y_val);
	y_plane[y * dstPitch + x] = y_val;

	px = src[y * srcAlignedWidth + x1];
	rgb_to_y(px.x, px.y, px.z, y_val);
	y_plane[y * dstPitch + x1] = y_val;

	px = src[y1 * srcAlignedWidth + x];
	rgb_to_y(px.x, px.y, px.z, y_val);
	y_plane[y1 * dstPitch + x] = y_val;
	
	px = src[y1 * srcAlignedWidth + x1];
	rgb_to_yuv(px.x, px.y, px.z, y_val, u_val, v_val);
	y_plane[y1 * dstPitch + x1] = y_val;

	const int uvPitch = dstPitch / 2;
	const int uvIndex = (y / 2) * uvPitch + (x / 2);

	u_plane[uvIndex] = u_val;
	v_plane[uvIndex] = v_val;
} 


template <typename T, bool formatNV12>
__global__ void RGB_to_NV12( T* src, int srcAlignedWidth, uint8_t* dst, int dstPitch, int width, int height )
{
	const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    printf("[%d,%d]\n",x,y);
	const int x1 = x + 1;
	const int y1 = y + 1;

	if( x1 >= width || y1 >= height )
		return;

	const int planeSize = height * dstPitch;
	
	uint8_t* y_plane = dst;
	uint8_t* u_plane;
	u_plane = y_plane + planeSize;
	

	T px;
	uint8_t y_val, u_val, v_val;

	px = src[y * srcAlignedWidth + x];
	rgb_to_y(px.x, px.y, px.z, y_val);
	y_plane[y * dstPitch + x] = y_val;

	px = src[y * srcAlignedWidth + x1];
	rgb_to_y(px.x, px.y, px.z, y_val);
	y_plane[y * dstPitch + x1] = y_val;

	px = src[y1 * srcAlignedWidth + x];
	rgb_to_y(px.x, px.y, px.z, y_val);
	y_plane[y1 * dstPitch + x] = y_val;
	
	px = src[y1 * srcAlignedWidth + x1];
	rgb_to_yuv(px.x, px.y, px.z, y_val, u_val, v_val);
	y_plane[y1 * dstPitch + x1] = y_val;

	const int uvPitch = dstPitch / 2;
	const int uvIndex = (y / 2) * uvPitch + (x / 2);

	if(formatNV12){
		u_plane[uvIndex*2+1] = u_val;
		u_plane[uvIndex*2] = v_val;
	}else{
		u_plane[uvIndex*2] = u_val;
		u_plane[uvIndex*2+1] = v_val;
	}
	
}


__device__ void YUV2RGB(uint32_t *yuvi, float *red, float *green, float *blue)
{
   

    // Prepare for hue adjustment
    /*
	 float luma, chromaCb, chromaCr;

	luma     = (float)yuvi[0];
    chromaCb = (float)((int)yuvi[1] - 512.0f);
    chromaCr = (float)((int)yuvi[2] - 512.0f);

    // Convert YUV To RGB with hue adjustment
    *red  = MUL(luma,     constHueColorSpaceMat[0]) +
            MUL(chromaCb, constHueColorSpaceMat[1]) +
            MUL(chromaCr, constHueColorSpaceMat[2]);
    *green= MUL(luma,     constHueColorSpaceMat[3]) +
            MUL(chromaCb, constHueColorSpaceMat[4]) +
            MUL(chromaCr, constHueColorSpaceMat[5]);
    *blue = MUL(luma,     constHueColorSpaceMat[6]) +
            MUL(chromaCb, constHueColorSpaceMat[7]) +
            MUL(chromaCr, constHueColorSpaceMat[8]);*/

	const float luma = float(yuvi[0]);
	const float u    = float(yuvi[1]) - 512.0f;
	const float v    = float(yuvi[2]) - 512.0f;

   /*R = Y + 1.140V
   G = Y - 0.395U - 0.581V
   B = Y + 2.032U*/

	/**green = luma + 1.140f * v;
	*blue  = luma - 0.395f * u - 0.581f * v;
	*red   = luma + 2.032f * u;*/

	*red    = luma + 1.140f * v;
	*green  = luma - 0.395f * u - 0.581f * v;
	*blue   = luma + 2.032f * u;
}


__device__ uint32_t RGBAPACK_8bit(float red, float green, float blue, uint32_t alpha)
{
    uint32_t ARGBpixel = 0;

    // Clamp final 10 bit results
    red   = min(max(red,   0.0f), 255.0f);
    green = min(max(green, 0.0f), 255.0f);
    blue  = min(max(blue,  0.0f), 255.0f);

    // Convert to 8 bit unsigned integers per color component
    ARGBpixel = ((((uint32_t)red)   << 24) | (((uint32_t)green) << 16) | (((uint32_t)blue)  <<  8) | (uint32_t)alpha);

    return  ARGBpixel;
}


__device__ uint32_t RGBAPACK_10bit(float red, float green, float blue, uint32_t alpha)
{
    uint32_t ARGBpixel = 0;

    // Clamp final 10 bit results
    red   = min(max(red,   0.0f), 1023.f);
    green = min(max(green, 0.0f), 1023.f);
	blue  = min(max(blue,  0.0f), 1023.f);
	uint32_t intRed = (uint32_t)red;
	intRed = intRed >> 2;

    // Convert to 8 bit unsigned integers per color component
    // ARGBpixel = ((((uint32_t)red   >> 2) << 24) |
    //              (((uint32_t)green >> 2) << 16) |
	// 			 (((uint32_t)blue  >> 2) <<  8) | (uint32_t)alpha);
	// ARGBpixel = ((((uint32_t)red   >> 2) << 24) |(((uint32_t)green >> 2) << 16) |(((uint32_t)blue  >> 2) <<  8) );
	// printf("[%d,%d] int red %d ,int green %d,blue %d",((uint32_t)red   >> 2),((uint32_t)green >> 2),((uint32_t)blue  >> 2),ARGBpixel);
	uint8_t * pRed =(uint8_t *) &ARGBpixel;
	*pRed=((uint32_t)red   >> 2);
	*(pRed+1)=((uint32_t)green >> 2);
	*(pRed+2)=((uint32_t)blue >> 2);

	printf("red is %d,green is %d,blue is %d,postion0-3,%d,%d,%d,%d\n",((uint32_t)red >> 2),((uint32_t)green >> 2),((uint32_t)blue >> 2),*(pRed),*(pRed+1),*(pRed+2),*(pRed+3));

    return  ARGBpixel;
}


__global__ void NV12ToARGB(uint32_t *srcImage,     size_t nSourcePitch,
	uint32_t *dstImage,     size_t nDestPitch,
	uint32_t width,         uint32_t height)
{
	int x, y;
	uint32_t yuv101010Pel[2];
	uint32_t processingPitch = ((width) + 63) & ~63;
	uint32_t dstImagePitch   = nDestPitch >> 2;
	uint8_t *srcImageU8     = (uint8_t *)srcImage;

	processingPitch = nSourcePitch;

	// Pad borders with duplicate pixels, and we multiply by 2 because we process 2 pixels per thread
	x = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
	y = blockIdx.y *  blockDim.y       +  threadIdx.y;

	if (x >= width)
	return; //x = width - 1;

	if (y >= height)
	return; // y = height - 1;

	// Read 2 Luma components at a time, so we don't waste processing since CbCr are decimated this way.
	// if we move to texture we could read 4 luminance values
	
	yuv101010Pel[0] = (srcImageU8[y * processingPitch + x    ]) << 2;
	yuv101010Pel[1] = (srcImageU8[y * processingPitch + x + 1]) << 2;

	
	uint32_t chromaOffset    = processingPitch * height;
	int y_chroma = y >> 1;

	if (y & 1)  // odd scanline ?
	{
	uint32_t chromaCb;
	uint32_t chromaCr;

	chromaCb = srcImageU8[chromaOffset + y_chroma * processingPitch + x    ];
	chromaCr = srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1];

	if (y_chroma < ((height >> 1) - 1)) // interpolate chroma vertically
	{
	chromaCb = (chromaCb + srcImageU8[chromaOffset + (y_chroma + 1) * processingPitch + x    ] + 1) >> 1;
	chromaCr = (chromaCr + srcImageU8[chromaOffset + (y_chroma + 1) * processingPitch + x + 1] + 1) >> 1;
	}

	yuv101010Pel[0] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE       + 2));
	yuv101010Pel[0] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

	yuv101010Pel[1] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE       + 2));
	yuv101010Pel[1] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
	}
	else
	{
	yuv101010Pel[0] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x    ] << (COLOR_COMPONENT_BIT_SIZE       + 2));
	yuv101010Pel[0] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1] << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

	yuv101010Pel[1] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x    ] << (COLOR_COMPONENT_BIT_SIZE       + 2));
	yuv101010Pel[1] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1] << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
	}

	// this steps performs the color conversion
	uint32_t yuvi[6];
	float red[2], green[2], blue[2];

	yuvi[0] = (yuv101010Pel[0] &   COLOR_COMPONENT_MASK);
	yuvi[1] = ((yuv101010Pel[0] >>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
	yuvi[2] = ((yuv101010Pel[0] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK);

	yuvi[3] = (yuv101010Pel[1] &   COLOR_COMPONENT_MASK);
	yuvi[4] = ((yuv101010Pel[1] >>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
	yuvi[5] = ((yuv101010Pel[1] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK);

	// YUV to RGB Transformation conversion
	YUV2RGB(&yuvi[0], &red[0], &green[0], &blue[0]);
	YUV2RGB(&yuvi[3], &red[1], &green[1], &blue[1]);
	

	// Clamp the results to RGBA
	dstImage[y * dstImagePitch + x     ] = RGBAPACK_10bit(red[0], green[0], blue[0], constAlpha);
	dstImage[y * dstImagePitch + x + 1 ] = RGBAPACK_10bit(red[1], green[1], blue[1], constAlpha);
	uint8_t * pRead =(uint8_t*) &dstImage[y * dstImagePitch + x     ];
	// if(x%4 ==0&&y%4==0){
	// 	printf("[%d,%d] red is %d,green is %d \n",x,y,pRead[0],pRead[1]);
	// }
	
}

void rgb2yuv(const char *src,uint8_t *dest,int width,int height){
    printf("rgb2yuv width %d,height %d\n",width,height);
    const dim3 block(32, 8);
	const dim3 grid(iDivUp(width, block.x * 2), iDivUp(height, block.y * 2));
    uchar3 * pChar3 = (uchar3 *) src;
     // Allocate the device input vector B
    uchar3 *nvPChar2 = NULL;
    cudaError_t err = cudaMalloc((void **)&nvPChar2, width*height*sizeof(uchar3));
    uint8_t *nvYuv = NULL;
    err = cudaMalloc((void **)&nvYuv, width*height*sizeof(uint8_t)*3/2);
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(nvPChar2, pChar3,  width*height*sizeof(uchar3), cudaMemcpyHostToDevice);
    RGB_to_YV12<uchar3, true><<<grid, block>>>(nvPChar2, width,nvYuv, width, width, height);
    err = cudaMemcpy(dest, nvYuv, width*height*3/2, cudaMemcpyDeviceToHost);
    err = cudaFree(nvPChar2);
    err = cudaFree(nvYuv);
    CHECK(cudaDeviceSynchronize());
}

void rgb2NV12(const char *src,uint8_t *dest,int width,int height){
    printf("rgb2yuv width %d,height %d\n",width,height);
    const dim3 block(32, 8);
	const dim3 grid(iDivUp(width, block.x * 2), iDivUp(height, block.y * 2));
    uchar3 * pChar3 = (uchar3 *) src;
     // Allocate the device input vector B
    uchar3 *nvPChar2 = NULL;
    cudaError_t err = cudaMalloc((void **)&nvPChar2, width*height*sizeof(uchar3));
    uint8_t *nvYuv = NULL;
    err = cudaMalloc((void **)&nvYuv, width*height*sizeof(uint8_t)*3/2);
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(nvPChar2, pChar3,  width*height*sizeof(uchar3), cudaMemcpyHostToDevice);
    RGB_to_NV12<uchar3, true><<<grid, block>>>(nvPChar2, width,nvYuv, width, width, height);
    err = cudaMemcpy(dest, nvYuv, width*height*3/2, cudaMemcpyDeviceToHost);
    err = cudaFree(nvPChar2);
    err = cudaFree(nvYuv);
    CHECK(cudaDeviceSynchronize());
}

bool nv12ColorspaceSetup = false;

// cudaNV12SetupColorspace
cudaError_t cudaNV12SetupColorspace( float hue = 0.0f )
{
	const float hueSin = sin(hue);
	const float hueCos = cos(hue);

	float hueCSC[9];

	const bool itu601 = false;

	if( itu601 /*CSC == ITU601*/)
	{
		//CCIR 601
		hueCSC[0] = 1.1644f;
		hueCSC[1] = hueSin * 1.5960f;
		hueCSC[2] = hueCos * 1.5960f;
		hueCSC[3] = 1.1644f;
		hueCSC[4] = (hueCos * -0.3918f) - (hueSin * 0.8130f);
		hueCSC[5] = (hueSin *  0.3918f) - (hueCos * 0.8130f);
		hueCSC[6] = 1.1644f;
		hueCSC[7] = hueCos *  2.0172f;
		hueCSC[8] = hueSin * -2.0172f;
	}
	else /*if(CSC == ITU709)*/
	{
		//CCIR 709
		hueCSC[0] = 1.0f;
		hueCSC[1] = hueSin * 1.57480f;
		hueCSC[2] = hueCos * 1.57480f;
		hueCSC[3] = 1.0;
		hueCSC[4] = (hueCos * -0.18732f) - (hueSin * 0.46812f);
		hueCSC[5] = (hueSin *  0.18732f) - (hueCos * 0.46812f);
		hueCSC[6] = 1.0f;
		hueCSC[7] = hueCos *  1.85560f;
		hueCSC[8] = hueSin * -1.85560f;
	}


	if( CUDA_FAILED(cudaMemcpyToSymbol(constHueColorSpaceMat, hueCSC, sizeof(float) * 9)) )
		return cudaErrorInvalidSymbol;

	uint32_t cudaAlpha = ((uint32_t)0xff);

	if( CUDA_FAILED(cudaMemcpyToSymbol(constAlpha, &cudaAlpha, sizeof(uint32_t))) )
		return cudaErrorInvalidSymbol;

	nv12ColorspaceSetup = true;
	return cudaSuccess;
}


// cudaNV12ToARGB32
void cudaNV12ToRGBA( char* srcDev, size_t srcPitch, char* destDev, size_t destPitch, size_t width, size_t height )
{
	// if( !srcDev || !destDev )
	// 	return cudaErrorInvalidDevicePointer;

	// if( srcPitch == 0 || destPitch == 0 || width == 0 || height == 0 )
	// 	return cudaErrorInvalidValue;

	if( !nv12ColorspaceSetup )
		cudaNV12SetupColorspace(0.0f);

	const dim3 blockDim(32,16,1);
	const dim3 gridDim((width+(2*blockDim.x-1))/(2*blockDim.x), (height+(blockDim.y-1))/blockDim.y, 1);
	char *nvRGBA = NULL;
    cudaError_t err = cudaMalloc((void **)&nvRGBA, width*height*sizeof(char)*4);
    char *nvNV12 = NULL;
    err = cudaMalloc((void **)&nvNV12, width*height*sizeof(char)*3/2);
    printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(nvNV12, srcDev,  width*height*sizeof(char)*3/2, cudaMemcpyHostToDevice);
	NV12ToARGB<<<gridDim, blockDim>>>( (uint32_t*)nvNV12, srcPitch, (uint32_t*)nvRGBA, destPitch, width, height );
	err = cudaMemcpy(destDev, nvRGBA, width*height*4, cudaMemcpyDeviceToHost);
    err = cudaFree(nvRGBA);
    err = cudaFree(nvNV12);
	CHECK(cudaDeviceSynchronize());
}
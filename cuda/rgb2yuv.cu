#include "rgb2yuv.cuh"
#include <stdint.h>
#include <cuda_runtime.h>

#define CHECK(res) { if(res != cudaSuccess){printf("Error ：%s:%d , ", __FILE__,__LINE__);   \
printf("code : %d , reason : %s \n", res,cudaGetErrorString(res));exit(-1);}}


/**
 * iDivUp
 */
 inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }


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
    err = cudaMemcpy(dest, nvPChar2, width*height*3/2, cudaMemcpyDeviceToHost);
    err = cudaFree(nvPChar2);
    err = cudaFree(nvYuv);
    CHECK(cudaDeviceSynchronize());
}
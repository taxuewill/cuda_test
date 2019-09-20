#include "transform.cuh"
#include <cuda_runtime.h>

#define CHECK(res) { if(res != cudaSuccess){printf("Error ：%s:%d , ", __FILE__,__LINE__);   \
printf("code : %d , reason : %s \n", res,cudaGetErrorString(res));exit(-1);}}


__global__ void cudaVectorAdd(const int *A,const int *B,int * C,int numElements)
{
    int i = threadIdx.x;
    //printf("cudaVectorAdd! %d \n",i);
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
        printf("cudaVectorAdd! A %d ,B %d, C %d\n",A[i],B[i],C[i]);
    }
}


void vectorAdd(const int *h_A,const int *h_B,int * h_C,int numElements)
{
    size_t size = numElements * sizeof(int);
     // Allocate the device input vector A
     int *d_A = NULL;
     cudaError_t err = cudaMalloc((void **)&d_A, size);
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
    // Allocate the device input vector B
    int *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    int *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);


    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    cudaVectorAdd<<<1, 1024>>>(d_A, d_B, d_C, numElements);

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
  // Free device global memory
  err = cudaFree(d_A);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaFree(d_B);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaFree(d_C);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

    CHECK(cudaDeviceSynchronize());

}

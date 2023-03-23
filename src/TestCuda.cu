#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "TestCuda.cuh"

__global__ void MatrixMultiply(glm::mat4* a, glm::mat4* b, glm::mat4* result)
{
	int x = blockIdx.x;
	int y = blockIdx.y;

    printf("Hello from block (%d, %d, %d), thread (%d, %d, %d), block dim (%d, %d, %d)\n",
        blockIdx.x, blockIdx.y, blockIdx.z,
        threadIdx.x, threadIdx.y, threadIdx.z,
        blockDim.x, blockDim.y, blockDim.z);
    if (x < 4 && y < 4)
    {
        (*result)[x][y] = 0.f;
        for (int i = 0; i < 4; ++i)
        {
            (*result)[x][y] += (*a)[x][i] * (*b)[i][y];
        }
    }
}

void executeCuda(glm::mat4* a, glm::mat4* b, glm::mat4* result)
{
    glm::mat4* dev_a = 0;
    glm::mat4* dev_b = 0;
    glm::mat4* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_a, sizeof(glm::mat4));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void**)&dev_b, sizeof(glm::mat4));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void**)&dev_c, sizeof(glm::mat4));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, sizeof(glm::mat4), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    cudaStatus = cudaMemcpy(dev_b, b, sizeof(glm::mat4), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    dim3 numBlocks(4, 4, 1);
    dim3 threadPerBlock(1, 1, 1);
    // Launch a kernel on the GPU with one thread for each element.
    MatrixMultiply <<< numBlocks, threadPerBlock >>> (dev_a, dev_b, dev_c);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "+ launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(result, dev_c, sizeof(glm::mat4), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
}
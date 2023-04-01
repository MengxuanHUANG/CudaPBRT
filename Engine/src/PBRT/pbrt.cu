#include "pbrt.h"

#include "ray.h"
#include "BVH/boundingBox.h"
#include "intersection.h"
#include "Shape/sphere.h"
//#include "bsdf.h"
//#include "bxdfs.h"

#include <GL/glew.h>
#include <cuda_gl_interop.h>

#ifdef CUDA_PBRT_DEBUG
#define CUDA_CHECK_ERROR(state, message) if(state != cudaSuccess) fprintf(stderr, message);
#else
#define CUDA_CHECK_ERROR(state, message) 
#endif

namespace CudaPBRT
{
    __global__ void CreateShapes(Shape** device_shapes, ShapeData* data, unsigned int* max_count)
    {
        int id = blockIdx.x;
        if (id >= *max_count)
        {
            return;
        }
        device_shapes[id] = new Sphere(data[id]);
    }
    
    __global__ void FreeShapes(Shape** device_shapes, unsigned int* max_count)
    {
        for (int i = 0; i < *max_count; ++i)
        {
            if (device_shapes[i])
            {
                delete device_shapes[i];
                device_shapes[i] = nullptr;
            }
        }
    }

    __global__ void NewPtr(TestCudaVirtual** tv)
    {
        (*tv) = new B();
        printf("new %X\n", (*tv));
        (*tv)->value.r = 255.f;
        (*tv)->value.g = -255.f;
    }

    __global__ void ReadPtr(TestCudaVirtual** tv)
    {
        //printf("read %X : (%f, %f, %f)\n",(*tv), (*tv)->value.x, (*tv)->value.y, (*tv)->value.z);
        printf("read %X : GetValue() {%f}\n", (*tv), (*tv)->GetValue());
    }

    __global__ void DeletePtr(TestCudaVirtual** tv)
    {
        printf("delete %X\n", (*tv));
        delete (*tv);
    }

    __global__ void Draw(PerspectiveCamera* camera, uchar4* img, Shape** shapes, unsigned int* shape_count)
    {
        int x = (blockIdx.x * blockDim.x) + threadIdx.x;
        int y = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (x >= camera->width || y >= camera->height) {
            return;
        }
        //printf("Hello from block (%d, %d, %d), thread (%d, %d, %d), block dim (%d, %d, %d)\n",
        //    blockIdx.x, blockIdx.y, blockIdx.z,
        //    threadIdx.x, threadIdx.y, threadIdx.z,
        //    blockDim.x, blockDim.y, blockDim.z);

        glm::vec2 ndc = 2.f * (glm::vec2(x, y) / glm::vec2(camera->width, camera->height));
        ndc.x = ndc.x - 1.f;
        ndc.y = 1.f - ndc.y;

        float aspect = camera->width / camera->height;

        // point in camera space
        glm::vec3 pCamera = glm::vec3(
            ndc.x * glm::tan(camera->fovy * 0.5f) * aspect,
            ndc.y * glm::tan(camera->fovy * 0.5f),
            1.f
        );

        Ray ray(glm::vec3(0), pCamera);

        ray.O = camera->position + ray.O.x * camera->right + ray.O.y * camera->up;
        ray.DIR = glm::normalize(ray.DIR.z * camera->forward +
                                 ray.DIR.y * camera->up +
                                 ray.DIR.x * camera->right);

        glm::vec3 color;

        // display normal
        //color = 0.5f * (ray.DIR + 1.f);

        color = glm::vec3(0.f);
        Intersection intersection;
        intersection.t = 10000.f;

        for (int i = 0; i < *shape_count; ++i)
        {
            Intersection it;

            if (shapes[i]->IntersectionP(ray, it) && it.t < intersection.t)
            {
                intersection = it;
                intersection.id = i;
            }
        }
        
        if (intersection.id >= 0)
        {
            color = 0.5f * (intersection.normal + 1.f);
        }
        
        // tone mapping
        //color = color / (1.f + color);

        // gammar correction
        //color = glm::pow(color, glm::vec3(1.f / 2.2f));

        img[y * camera->width + x].x = static_cast<int>(glm::mix(0.f, 255.f, color.x));
        img[y * camera->width + x].y = static_cast<int>(glm::mix(0.f, 255.f, color.y));
        img[y * camera->width + x].z = static_cast<int>(glm::mix(0.f, 255.f, color.z));
        img[y * camera->width + x].w = 255;
    }

    CudaPathTracer::CudaPathTracer()
    {

    }

    CudaPathTracer::~CudaPathTracer()
    {

    }

    void CudaPathTracer::InitCuda(PerspectiveCamera& camera, int device)
    {
        // set basic properties
        width = camera.width;
        height = camera.height;

        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaError_t cudaStatus = cudaSetDevice(device);;
        CUDA_CHECK_ERROR(cudaStatus, "cudaSetDevice failed!");

        glGenTextures(1, &m_DisplayImage);
        glBindTexture(GL_TEXTURE_2D, m_DisplayImage);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);

        // create rendered image on cpu
        host_image = new uchar4[width * height];

        // Create cuda device pointers
        // Allocate GPU buffers for three vectors (two input, one output).
        cudaStatus = cudaMalloc((void**)&device_camera, sizeof(PerspectiveCamera));
        CUDA_CHECK_ERROR(cudaStatus, "cudaMalloc failed!");

        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(device_camera, &camera, sizeof(PerspectiveCamera), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR(cudaStatus, "cudaMemcpy host to device failed!");

        cudaStatus = cudaMalloc((void**)&device_image, sizeof(uchar4) * width * height);
        CUDA_CHECK_ERROR(cudaStatus, "cudaMalloc failed!");
    }

    void CudaPathTracer::CreateShapesOnCuda(std::vector<ShapeData>& shapeData)
    {
        cudaError_t cudaStatus;

        ShapeData* device_shapeData;
        unsigned int max_count = shapeData.size();

        cudaStatus = cudaMalloc((void**)&device_shape_count, sizeof(unsigned int));
        CUDA_CHECK_ERROR(cudaStatus, "cudaMalloc failed!");

        cudaStatus = cudaMemcpy(device_shape_count, &max_count, sizeof(unsigned int), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR(cudaStatus, "cudaMemcpy host to device failed!");

        cudaStatus = cudaMalloc((void**)&device_shapeData, sizeof(ShapeData) * max_count);
        CUDA_CHECK_ERROR(cudaStatus, "cudaMalloc failed!");

        cudaStatus = cudaMemcpy(device_shapeData, shapeData.data(), sizeof(ShapeData) * max_count, cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR(cudaStatus, "cudaMemcpy host to device failed!");

        cudaStatus = cudaMalloc((void**)&device_shapes, sizeof(Shape*) * max_count);
        CUDA_CHECK_ERROR(cudaStatus, "cudaMalloc failed!");

        // Launch a kernel on the GPU with one thread for each element.
        CreateShapes <<< max_count, 1 >>> (device_shapes, device_shapeData, device_shape_count);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        CUDA_CHECK_ERROR(cudaStatus, "cuda launch failed!");

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error %s\n", cudaGetErrorString(cudaStatus));
        }

        if (device_shapeData != nullptr)
        {
            cudaFree(device_shapeData);
        }
    }

    void CudaPathTracer::FreeShapesOnCuda()
    {
        cudaError_t cudaStatus;

        // Launch a kernel on the GPU with one thread for each element.
        FreeShapes <<<1, 1>>> (device_shapes, device_shape_count);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        CUDA_CHECK_ERROR(cudaStatus, "cuda launch failed!");

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error %s\n", cudaGetErrorString(cudaStatus));
        }
    }

    void CudaPathTracer::FreeCuda()
    {
        if (device_camera != nullptr)
        {
            cudaFree(device_camera);
            device_camera = nullptr;
        }
        if (device_image != nullptr)
        {
            cudaFree(device_image);
            device_image = nullptr;
        }
        if (device_shapes != nullptr)
        {
            cudaFree(device_shapes);
            device_shapes = nullptr;
        }
        if (device_shape_count != nullptr)
        {
            cudaFree(device_shape_count);
        }

        if (host_image)
        {
            delete[] host_image;
        }
        if (m_DisplayImage)
        {
            glDeleteTextures(1, &m_DisplayImage);
        }
    }

    void CudaPathTracer::Run()
    {
        cudaError_t cudaStatus;

        dim3 numBlocks(UpperBinary(width >> 4), UpperBinary(height >> 4), 1);
        dim3 threadPerBlock(16, 16, 1);

        // Launch a kernel on the GPU with one thread for each element.
        Draw <<< numBlocks, threadPerBlock >>> (device_camera, device_image, device_shapes, device_shape_count);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        CUDA_CHECK_ERROR(cudaStatus, "cuda launch failed!");

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error %s\n", cudaGetErrorString(cudaStatus));
        }

        // Copy rendered result to CPU.
        cudaStatus = cudaMemcpy(host_image, device_image, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERROR(cudaStatus, "cudaMemcpy from device to host failed!");

        // pass render result to glTexture2D
        glBindTexture(GL_TEXTURE_2D, m_DisplayImage);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)host_image);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void CudaPathTracer::UpdateCamera(PerspectiveCamera& camera)
    {
        // Copy input vectors from host memory to GPU buffers.
        cudaError_t cudaStatus = cudaMemcpy(device_camera, &camera, sizeof(PerspectiveCamera), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR(cudaStatus, "cudaMemcpy host to device failed!");
    }
}
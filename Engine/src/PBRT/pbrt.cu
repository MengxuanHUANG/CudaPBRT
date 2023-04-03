#include "pbrt.h"

#include "ray.h"
#include "BVH/boundingBox.h"
#include "intersection.h"
#include "Shape/sphere.h"
#include "Shape/square.h"
#include "Shape/cube.h"

#include "Spectrum.h"
//#include "bsdf.h"
//#include "bxdfs.h"

#include <GL/glew.h>
#include <cuda_gl_interop.h>

namespace CudaPBRT
{
    CPU_GPU Shape* CreateShape(const ShapeData& data)
    {
        switch (data.type)
        {
        case ShapeType::Sphere:
            return new Sphere(data);
        case ShapeType::Cube:
            return new Cube(data);
        case ShapeType::Square:
            return new Square(data);
        default:
            printf("Unknown ShapeType!\n");
            return nullptr;
        }
    }
    
    CPU_GPU Ray CastRay(const PerspectiveCamera& camera, const glm::vec2& p)
    {
        glm::vec2 ndc = 2.f * p / glm::vec2(camera.width, camera.height);
        ndc.x = ndc.x - 1.f;
        ndc.y = 1.f - ndc.y;

        float aspect = static_cast<float>(camera.width) / static_cast<float>(camera.height);

        // point in camera space
        float radian = glm::radians(camera.fovy * 0.5f);
        glm::vec3 pCamera = glm::vec3(
            ndc.x * glm::tan(radian) * aspect,
            ndc.y * glm::tan(radian),
            1.f
        );

        Ray ray(glm::vec3(0), pCamera);

        ray.O = camera.position + ray.O.x * camera.right + ray.O.y * camera.up;
        ray.DIR = glm::normalize(ray.DIR.z * camera.forward +
                                 ray.DIR.y * camera.up +
                                 ray.DIR.x * camera.right);

        return ray;
    }

    CPU_GPU void writePixel(uchar4& pixel, const Spectrum& radiance)
    {
        // tone mapping
        Spectrum color = radiance / (1.f + radiance);

        // gammar correction
        color = glm::pow(color, Spectrum(1.f / 2.2f));

        pixel.x = static_cast<int>(glm::mix(0.f, 255.f, color.x));
        pixel.y = static_cast<int>(glm::mix(0.f, 255.f, color.y));
        pixel.z = static_cast<int>(glm::mix(0.f, 255.f, color.z));
        pixel.w = 255;
    }

    __global__ void CreateShapes(Shape** device_shapes, ShapeData* data, unsigned int* max_count)
    {
        int id = blockIdx.x;
        if (id >= *max_count)
        {
            return;
        }

        device_shapes[id] = CreateShape(data[id]);
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

    __global__ void Draw(PerspectiveCamera* camera, uchar4* img, Shape** shapes, unsigned int* shape_count)
    {
        int x = (blockIdx.x * blockDim.x) + threadIdx.x;
        int y = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (x >= camera->width || y >= camera->height) {
            return;
        }

        Ray ray = CastRay(*camera, {x, y});

        Spectrum radiance(0.f);

        Intersection intersection;
        intersection.t = 10000.f;

        // TODO: use BVH for intersection testing
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
            radiance = 0.5f * (intersection.normal + 1.f);
        }
        
        writePixel(img[y * camera->width + x], radiance);
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
        cudaSetDevice(device);;
        CUDA_CHECK_ERROR();

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
        cudaMalloc((void**)&device_camera, sizeof(PerspectiveCamera));
        CUDA_CHECK_ERROR();

        // Copy input vectors from host memory to GPU buffers.
        cudaMemcpy(device_camera, &camera, sizeof(PerspectiveCamera), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();

        cudaMalloc((void**)&device_image, sizeof(uchar4) * width * height);
        CUDA_CHECK_ERROR();
    }

    void CudaPathTracer::CreateShapesOnCuda(std::vector<ShapeData>& shapeData)
    {
        ShapeData* device_shapeData;
        unsigned int max_count = shapeData.size();

        cudaMalloc((void**)&device_shape_count, sizeof(unsigned int));
        CUDA_CHECK_ERROR();

        cudaMemcpy(device_shape_count, &max_count, sizeof(unsigned int), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();

        cudaMalloc((void**)&device_shapeData, sizeof(ShapeData) * max_count);
        CUDA_CHECK_ERROR();

        cudaMemcpy(device_shapeData, shapeData.data(), sizeof(ShapeData) * max_count, cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();

        cudaMalloc((void**)&device_shapes, sizeof(Shape*) * max_count);
        CUDA_CHECK_ERROR();

        // Launch a kernel on the GPU with one thread for each element.
        CreateShapes<<< max_count, 1 >>> (device_shapes, device_shapeData, device_shape_count);

        // cudaDeviceSynchronize waits for the kernel to finish
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR();

        CUDA_FREE(device_shapeData);
    }

    void CudaPathTracer::FreeShapesOnCuda()
    {
        FreeShapes <<<1, 1>>> (device_shapes, device_shape_count);
        CUDA_CHECK_ERROR();
    }

    void CudaPathTracer::FreeCuda()
    {
        FreeShapesOnCuda();

        CUDA_FREE(device_camera);
        CUDA_FREE(device_image);
        CUDA_FREE(device_shapes);
        CUDA_FREE(device_shape_count);

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
        KernalConfig drawConfig({width, height, 1}, {4, 4, 0});

        //glm::ivec2 blockSize(5, 5);
        //dim3 numBlocks(UpperBinary(width >> blockSize.x), UpperBinary(height >> blockSize.y), 1);
        //dim3 threadPerBlock(BIT(blockSize.x), BIT(blockSize.y), 1);

        // draw color to pixels
        Draw <<< drawConfig.numBlocks, drawConfig.threadPerBlock >>> (device_camera, device_image, device_shapes, device_shape_count);

        // wait GPU to finish computation
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR();

        // Copy rendered result to CPU.
        cudaMemcpy(host_image, device_image, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERROR();

        // pass render result to glTexture2D
        glBindTexture(GL_TEXTURE_2D, m_DisplayImage);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)host_image);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void CudaPathTracer::UpdateCamera(PerspectiveCamera& camera)
    {
        // Copy input vectors from host memory to GPU buffers.
        cudaMemcpy(device_camera, &camera, sizeof(PerspectiveCamera), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();
    }
}
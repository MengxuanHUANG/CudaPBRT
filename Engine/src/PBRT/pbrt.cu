#include "pbrt.h"

#include "ray.h"
#include "BVH/boundingBox.h"
#include "intersection.h"
#include "Shape/sphere.h"
#include "Shape/square.h"
#include "Shape/cube.h"

#include "spectrum.h"
#include "Material/diffuseMaterial.h"

#include "Sampler/rng.h"
#include "Light/light.h"

#include <GL/glew.h>
#include <cuda_gl_interop.h>

namespace CudaPBRT
{
    CPU_GPU Shape* Create(const ShapeData& data)
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
    
    CPU_GPU Material* Create(const MaterialData& data)
    {
        switch (data.type)
        {
        case MaterialType::DiffuseReflection:
            return new DiffuseMaterial(data);
        default:
            printf("Unknown MaterialType!\n");
            return nullptr;
        }
    }

    CPU_GPU Light* Create(const LightData& data)
    {
        switch (data.type)
        {
        case LightType::ShapeLight:
            return new ShapeLight(data);
        default:
            printf("Unknown LightType!\n");
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

    CPU_GPU void writePixel(int iteration, float3& hdr_pixel, uchar4& pixel, const Spectrum& radiance)
    {
        glm::vec3 color(radiance);
        glm::vec3 preColor = glm::vec3(hdr_pixel.x, hdr_pixel.y, hdr_pixel.z);

        color = glm::mix(preColor, color, 1.f / float(iteration));
        
        hdr_pixel.x = color.x;
        hdr_pixel.y = color.y;
        hdr_pixel.z = color.z;

        // tone mapping
        color = color / (1.f + color);

        // gammar correction
        color = glm::pow(color, glm::vec3(1.f / 2.2f));

        color = glm::mix(glm::vec3(0.f), glm::vec3(255.f), color);

        color = glm::clamp(color, glm::vec3(0), glm::vec3(255));

        pixel.x = static_cast<int>(color.r);
        pixel.y = static_cast<int>(color.g);
        pixel.z = static_cast<int>(color.b);
        pixel.w = 255;
    }

    template<typename T, typename DataType>
    __global__ void CreateArray(T** device_array, DataType* data, size_t* max_count)
    {
        int id = blockIdx.x;
        if (id >= (*max_count))
        {
            return;
        }
        
        device_array[id] = Create(data[id]);
    }

    template<typename T>
    __global__ void FreeArray(T** device_array, size_t* max_count)
    {
        if (max_count == nullptr)
        {
            return;
        }

        for (int i = 0; i < *max_count; ++i)
        {
            if (device_array[i])
            {
                delete device_array[i];
                device_array[i] = nullptr;
            }
        }
    }
    

    template<typename T, typename DataType>
    void CreateArrayOnCude<T, DataType>(T**& dev_array, size_t*& dev_count, std::vector<DataType>& host_data)
    {
        DataType* device_data;
        size_t max_count = host_data.size();

        cudaMalloc((void**)&dev_count, sizeof(size_t));
        CUDA_CHECK_ERROR();

        cudaMemcpy(dev_count, &max_count, sizeof(size_t), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();

        cudaMalloc((void**)&device_data, sizeof(DataType) * max_count);
        CUDA_CHECK_ERROR();

        cudaMemcpy(device_data, host_data.data(), sizeof(DataType) * max_count, cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();

        cudaMalloc((void**)&dev_array, sizeof(T*) * max_count);
        CUDA_CHECK_ERROR();

        // Launch a kernel on the GPU with one thread for each element.
        KernalConfig createConfig({ max_count, 1, 1 }, { 0, 0, 0 });
        CreateArray<T, DataType> << < createConfig.numBlocks, createConfig.threadPerBlock >> > (dev_array, device_data, dev_count);

        // cudaDeviceSynchronize waits for the kernel to finish
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR();

        CUDA_FREE(device_data);
    }

    template void CreateArrayOnCude<Light, LightData>(Light**& dev_array, size_t*& dev_count, std::vector<LightData>& data);
    template void CreateArrayOnCude<Shape, ShapeData>(Shape**& dev_array, size_t*& dev_count, std::vector<ShapeData>& data);
    template void CreateArrayOnCude<Material, MaterialData>(Material**& dev_array, size_t*& dev_count, std::vector<MaterialData>& data);

    template<typename T>
    void FreeArrayOnCuda(T**& device_array, size_t*& count)
    {
        if (count == nullptr || device_array == nullptr)
        {
            return;
        }

        KernalConfig freeConfig({ 1, 1, 1 }, { 0, 0, 0 });
        FreeArray<T> << <freeConfig.numBlocks, freeConfig.threadPerBlock >> > (device_array, count);
        CUDA_CHECK_ERROR();

        CUDA_FREE(device_array);
        CUDA_FREE(count);
        CUDA_CHECK_ERROR();
    }

    template void FreeArrayOnCuda(Shape**& device_array, size_t*& count);
    template void FreeArrayOnCuda(Material**& device_array, size_t*& count);
    template void FreeArrayOnCuda(Light**& device_array, size_t*& count);

    __global__ void Draw(int* iteration, PerspectiveCamera* camera, uchar4* img, float3* hdr_img, Shape** shapes, size_t* shape_count, Light** lights, size_t* light_count, Material** materials)
    {
        int x = (blockIdx.x * blockDim.x) + threadIdx.x;
        int y = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (x >= camera->width || y >= camera->height) {
            return;
        }
        
        int index = x + (y * camera->width);
        
        CudaRNG rng(*iteration, index, 1);

        Ray ray = CastRay(*camera, {x + rng.rand(), y + rng.rand() });

        Spectrum radiance(0.f);
        Spectrum throughput(1.f);
        
        Intersection shape_intersection, light_intersection;

        int depth = 0;
        while (depth++ < MaxDepth)
        {
            // find itersection
            // TODO: use BVH for intersection testing
            shape_intersection.t = CudaPBRT::FloatMax;
            light_intersection.t = CudaPBRT::FloatMax;
            shape_intersection.id = -1;
            light_intersection.id = -1;

            for (int i = 0; i < (*light_count); ++i)
            {
                Intersection it;
                if (lights[i]->IntersectionP(ray, it) && it.t > 0.f && it.t < light_intersection.t)
                {
                    light_intersection = it;
                    light_intersection.id = i;
                }
            }
            for (int i = 0; i < (*shape_count); ++i)
            {
                Intersection it;
                if (shapes[i]->IntersectionP(ray, it) && it.t > 0.f && it.t < shape_intersection.t)
                {
                    shape_intersection = it;
                    shape_intersection.id = i;
                    shape_intersection.material_id = shapes[i]->material_id;
                }
            }

            if (light_intersection.id >= 0)
            {
                if (shape_intersection.id < 0 || (shape_intersection.id >= 0 && shape_intersection.t > light_intersection.t))
                {
                    // hit light source
                    throughput *= lights[light_intersection.id]->GetLe();
                    radiance += throughput;
                    break;
                }
            }
            else if (shape_intersection.id >= 0)
            {
                //radiance += materials[shape_intersection.material_id]->GetAlbedo();
                //radiance += 0.5f * (shape_intersection.normal + 1.f);
                //break;
                glm::vec3 point = ray * shape_intersection.t;
                BSDF& bsdf = materials[shape_intersection.material_id]->GetBSDF();

                glm::vec3 normal = glm::normalize(shape_intersection.normal);
                normal = materials[shape_intersection.material_id]->GetNormal(normal);

                BSDFSample bsdfSample = bsdf.Sample_f(materials[shape_intersection.material_id]->GetAlbedo(), -ray.DIR, normal, {rng.rand(), rng.rand()});

                if (bsdfSample.pdf == 0.f || glm::length(bsdfSample.f) == 0.f)
                {
                    break;
                }

                throughput *= bsdfSample.f * glm::abs(glm::dot(bsdfSample.wiW, normal)) / bsdfSample.pdf;
                
                ray = Ray::SpawnRay(point, bsdfSample.wiW);
            }
            else
            {
                break;
            }
        }

        writePixel(*iteration, hdr_img[y * camera->width + x], img[y * camera->width + x], radiance);
    }

    CudaPathTracer::CudaPathTracer()
    {

    }

    CudaPathTracer::~CudaPathTracer()
    {
        FreeCuda();
    }

    void CudaPathTracer::InitCuda(PerspectiveCamera& camera, int device)
    {
        // set basic properties
        width = camera.width;
        height = camera.height;
        
        m_Iteration = 1;

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
        cudaMalloc((void**)&device_iteration, sizeof(int));
        CUDA_CHECK_ERROR();     

        cudaMalloc((void**)&device_camera, sizeof(PerspectiveCamera));
        CUDA_CHECK_ERROR();

        // Copy input vectors from host memory to GPU buffers.
        cudaMemcpy(device_camera, &camera, sizeof(PerspectiveCamera), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();

        cudaMalloc((void**)&device_hdr_image, sizeof(float3) * width * height);
        CUDA_CHECK_ERROR();

        cudaMalloc((void**)&device_image, sizeof(uchar4) * width * height);
        CUDA_CHECK_ERROR();
    }

    void CudaPathTracer::FreeCuda()
    {
        FreeArrayOnCuda<Shape>(device_shapes, device_shape_count);
        FreeArrayOnCuda<Material>(device_materials, device_material_count);
        FreeArrayOnCuda<Light>(device_lights, device_light_count);

        CUDA_FREE(device_camera);
        CUDA_FREE(device_image);
        CUDA_FREE(device_hdr_image);
        CUDA_FREE(device_iteration);

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

        cudaMemcpy(device_iteration, &m_Iteration, sizeof(int), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();

        KernalConfig drawConfig({width, height, 1}, {3, 3, 0});

        //glm::ivec2 blockSize(5, 5);
        //dim3 numBlocks(UpperBinary(width >> blockSize.x), UpperBinary(height >> blockSize.y), 1);
        //dim3 threadPerBlock(BIT(blockSize.x), BIT(blockSize.y), 1);

        // draw color to pixels
        Draw <<< drawConfig.numBlocks, drawConfig.threadPerBlock >>> (device_iteration, device_camera, device_image, device_hdr_image,
                                                                      device_shapes, device_shape_count, 
                                                                      device_lights, device_light_count,
                                                                      device_materials);

        ++m_Iteration;

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
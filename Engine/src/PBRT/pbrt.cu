#include "pbrt.h"

#include "spectrum.h"
#include "ray.h"
#include "pathSegment.h"
#include "scene.h"

#include "BVH/boundingBox.h"
#include "intersection.h"
#include "Shape/sphere.h"
#include "Shape/square.h"
#include "Shape/cube.h"

#include "Material/diffuseMaterial.h"
#include "Sampler/rng.h"
#include "Light/light.h"

#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include <thrust/remove.h>
#include <thrust/sort.h>

namespace CudaPBRT
{
	struct CompactTerminatedPaths {
		CPU_GPU bool operator() (const PathSegment& segment) {
			return !(segment.pixelId >= 0 && segment.IsEnd());
		}
	};

	struct RemoveInvalidPaths {
		CPU_GPU bool operator() (const PathSegment& segment) {
			return segment.pixelId < 0 || segment.IsEnd();
		}
	};

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

	INLINE CPU_GPU void writePixel(int iteration, float3& hdr_pixel, uchar4& pixel, const Spectrum& radiance)
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
	__global__ void CreateArray(T** device_array, DataType* data, size_t max_count)
	{
		int id = blockIdx.x;
		if (id >= max_count)
		{
			return;
		}
		
		device_array[id] = Create(data[id]);
	}

	template<typename T>
	__global__ void FreeArray(T** device_array, size_t max_count)
	{
		for (int i = 0; i < max_count; ++i)
		{
			if (device_array[i])
			{
				delete device_array[i];
				device_array[i] = nullptr;
			}
		}
	}
	
	template<typename T, typename DataType>
	void CreateArrayOnCude<T, DataType>(T**& dev_array, size_t& count, std::vector<DataType>& host_data)
	{
		DataType* device_data;
		count = host_data.size();

		cudaMalloc((void**)&device_data, sizeof(DataType) * count);
		CUDA_CHECK_ERROR();

		cudaMemcpy(device_data, host_data.data(), sizeof(DataType) * count, cudaMemcpyHostToDevice);
		CUDA_CHECK_ERROR();

		cudaMalloc((void**)&dev_array, sizeof(T*) * count);
		CUDA_CHECK_ERROR();

		// Launch a kernel on the GPU with one thread for each element.
		KernalConfig createConfig({ count, 1, 1 }, { 0, 0, 0 });
		CreateArray<T, DataType> << < createConfig.numBlocks, createConfig.threadPerBlock >> > (dev_array, device_data, count);

		// cudaDeviceSynchronize waits for the kernel to finish
		cudaDeviceSynchronize();
		CUDA_CHECK_ERROR();

		CUDA_FREE(device_data);
	}

	template void CreateArrayOnCude<Light, LightData>(Light**& dev_array, size_t& dev_count, std::vector<LightData>& data);
	template void CreateArrayOnCude<Shape, ShapeData>(Shape**& dev_array, size_t& dev_count, std::vector<ShapeData>& data);
	template void CreateArrayOnCude<Material, MaterialData>(Material**& dev_array, size_t& dev_count, std::vector<MaterialData>& data);

	template<typename T>
	void FreeArrayOnCuda(T**& device_array, size_t count)
	{
		if (count == 0 || device_array == nullptr)
		{
			return;
		}

		KernalConfig freeConfig({ 1, 1, 1 }, { 0, 0, 0 });
		FreeArray<T> << <freeConfig.numBlocks, freeConfig.threadPerBlock >> > (device_array, count);
		CUDA_CHECK_ERROR();

		CUDA_FREE(device_array);
		CUDA_CHECK_ERROR();
	}

	template void FreeArrayOnCuda(Shape**& device_array, size_t count);
	template void FreeArrayOnCuda(Material**& device_array, size_t count);
	template void FreeArrayOnCuda(Light**& device_array, size_t count);

	__global__ void GlobalCastRayFromCamera(int* iteration, PerspectiveCamera* camera, PathSegment* pathSegment)
	{
		int x = (blockIdx.x * blockDim.x) + threadIdx.x;
		int y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x >= camera->width || y >= camera->height) {
			return;
		}

		int index = (y * camera->width) + x;

		PathSegment& segment = pathSegment[index];

		segment.Reset();

		CudaRNG rng(*iteration, index, 1);
		segment.ray = CastRay(*camera, { x + rng.rand(), y + rng.rand() });
		segment.pixelId = index;
	}

	__global__ void GlobalSceneIntersection(int max_index, PathSegment* pathSegment, Scene scene)
	{
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= max_index)
		{
			return;
		}
		
		PathSegment& segment = pathSegment[index];
		Ray& ray = segment.ray;
		Intersection& seg_int = segment.intersection;
		
		seg_int.Reset();

		if (!scene.IntersectionNaive(ray, seg_int))
		{
			segment.End();
		}
	}

	__global__ void GlobalNaiveLi(int* iteration, int max_index, PathSegment* pathSegment, Scene scene)
	{
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index >= max_index || pathSegment[index].IsEnd())
		{
			return;
		}

		PathSegment& segment = pathSegment[index];
		Intersection& intersection = segment.intersection;
		Ray& ray = segment.ray;
		
		if (intersection.isLight)
		{
			// hit light source
			segment.throughput *= scene.lights[intersection.id]->GetLe();
			segment.radiance += segment.throughput;
			segment.End();
		}
		else
		{
			Material* material = scene.materials[intersection.material_id];
			CudaRNG rng(*iteration, index, 4 + segment.depth * 7);

			BSDF& bsdf = material->GetBSDF();

			glm::vec3 normal = glm::normalize(intersection.normal);
			normal = material->GetNormal(normal);

			BSDFSample bsdfSample = bsdf.Sample_f(material->GetAlbedo(), -ray.DIR, normal, { rng.rand(), rng.rand() });

			if (bsdfSample.pdf == 0.f || glm::length(bsdfSample.f) == 0.f)
			{
				segment.End();
			}
			else
			{
				segment.throughput *= bsdfSample.f * glm::abs(glm::dot(bsdfSample.wiW, normal)) / bsdfSample.pdf;
				segment.ray = Ray::SpawnRay(ray * intersection.t, bsdfSample.wiW);
				segment.depth += 1;
			}
		}
	}
	
	__global__ void GlobalDirectLi(int* iteration, int max_index, PathSegment* pathSegment, Scene scene)
	{
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index >= max_index || pathSegment[index].IsEnd())
		{
			return;
		}

		PathSegment& segment = pathSegment[index];
		Intersection& intersection = segment.intersection;
		Ray& ray = segment.ray;

		if (intersection.isLight)
		{
			// hit light source
			segment.throughput *= scene.lights[intersection.id]->GetLe();
			segment.radiance += segment.throughput;
			segment.End();
		}
		else
		{
			Material* material = scene.materials[intersection.material_id];
			glm::vec3 normal = glm::normalize(intersection.normal);
			normal = material->GetNormal(normal);

			CudaRNG rng(*iteration, index, 4 + segment.depth * 7);
			LightSample sample;
			if (scene.Sample_Li(rng.rand(), {rng.rand(), rng.rand()}, ray * intersection.t, normal, sample))
			{
				segment.throughput *= material->GetAlbedo() * sample.Le * glm::abs(glm::dot(sample.wiW, normal)) / sample.pdf;

				segment.radiance += segment.throughput;
			}
			segment.End();
		}
	}

	__global__ void GlobalWritePixel(int* iteration, int max_index, PathSegment* pathSegment, uchar4* img, float3* hdr_img)
	{
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		
		if (index >= max_index)
		{
			return;
		}
		PathSegment& segment = pathSegment[index];
		int& pixelId = segment.pixelId;

		writePixel(*iteration, hdr_img[pixelId], img[pixelId], segment.radiance);
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

		cudaMalloc((void**)&device_image, sizeof(uchar4) * width * height);
		CUDA_CHECK_ERROR();

		cudaMalloc((void**)&device_hdr_image, sizeof(float3) * width * height);
		CUDA_CHECK_ERROR();

		cudaMalloc((void**)&device_pathSegment, sizeof(PathSegment) * width * height);
		CUDA_CHECK_ERROR();
		
		cudaMalloc((void**)&device_terminatedPathSegment, sizeof(PathSegment) * width * height);
		CUDA_CHECK_ERROR();

		devPathsThr = thrust::device_ptr<PathSegment>(device_pathSegment);
		devTerminatedPathsThr = thrust::device_ptr<PathSegment>(device_terminatedPathSegment);

		// Copy input vectors from host memory to GPU buffers.
		cudaMemcpy(device_camera, &camera, sizeof(PerspectiveCamera), cudaMemcpyHostToDevice);
		CUDA_CHECK_ERROR();
	}

	void CudaPathTracer::FreeCuda()
	{
		CUDA_FREE(device_camera);
		CUDA_FREE(device_image);
		CUDA_FREE(device_hdr_image);
		CUDA_FREE(device_iteration);
		CUDA_FREE(device_pathSegment);
		CUDA_FREE(device_terminatedPathSegment);

		if (host_image)
		{
			delete[] host_image;
		}
		if (m_DisplayImage)
		{
			glDeleteTextures(1, &m_DisplayImage);
		}
	}

	void CudaPathTracer::Run(Scene* scene)
	{
		int it = m_Iteration++;
		int max_count = width * height;
		
		auto devTerminatedThr = devTerminatedPathsThr;

		cudaMemcpy(device_iteration, &it, sizeof(int), cudaMemcpyHostToDevice);
		CUDA_CHECK_ERROR();
		
		// cast ray from camera
		KernalConfig CamConfig({ width, height, 1 }, { 3, 3, 0 });
		GlobalCastRayFromCamera << < CamConfig.numBlocks, CamConfig.threadPerBlock >> > (device_iteration, device_camera, device_pathSegment);
		cudaDeviceSynchronize();
		CUDA_CHECK_ERROR();

		int depth = 0;
		while (max_count > 0 && depth++ < CudaPBRT::PathMaxDepth)
		{
			KernalConfig intersectionConfig({ max_count, 1, 1 }, { 7, 0, 0 });

			// intersection
			GlobalSceneIntersection << < intersectionConfig.numBlocks, intersectionConfig.threadPerBlock >> > (max_count, device_pathSegment, *scene);
			cudaDeviceSynchronize();
			CUDA_CHECK_ERROR();

			//thrust::sort(devPathsThr, devPathsThr + max_count);

			KernalConfig throughputConfig({ max_count, 1, 1 }, { 7, 0, 0 });
			
			//GlobalNaiveLi << <throughputConfig.numBlocks, throughputConfig.threadPerBlock >> > (device_iteration, max_count, device_pathSegment, *scene);
			
			GlobalDirectLi << <throughputConfig.numBlocks, throughputConfig.threadPerBlock >> > (device_iteration, max_count, device_pathSegment, *scene);

			cudaDeviceSynchronize();
			CUDA_CHECK_ERROR();

			devTerminatedThr = thrust::remove_copy_if(devPathsThr, devPathsThr + max_count, devTerminatedThr, CompactTerminatedPaths());
			auto end = thrust::remove_if(devPathsThr, devPathsThr + max_count, RemoveInvalidPaths());
			
			max_count = end - devPathsThr;
		}
		int numContributing = devTerminatedThr.get() - device_terminatedPathSegment;
		KernalConfig pixelConfig({ numContributing, 1, 1 }, { 7, 0, 0 });
		
		GlobalWritePixel << <pixelConfig.numBlocks, pixelConfig.threadPerBlock >> > (device_iteration, numContributing, device_terminatedPathSegment,
																					 device_image, device_hdr_image);
		
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
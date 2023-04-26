#include "pbrt.h"

#include "spectrum.h"

#include "pathSegment.h"
#include "scene.h"

#include "texture.h"

#include "BVH/boundingBox.h"
#include "intersection.h"
#include "Shape/sphere.h"
#include "Shape/square.h"
#include "Shape/cube.h"
#include "Shape/triangle.h"

#include "Material/diffuseMaterial.h"
#include "Material/specularMaterial.h"
#include "Material/glass.h"
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
		case ShapeType::Triangle:
			return new Triangle(data);
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
		case MaterialType::SpecularReflection:
			return new SpecularMaterial(data);
		case MaterialType::SpecularTransmission:
			return new Glass(data);
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

	GPU_ONLY float4 ReadTexture(const CudaTexObj& tex_obj, const glm::vec2& uv)
	{
		return tex2D<float4>(tex_obj, uv[0], uv[1]);
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
		count = host_data.size();
		if (count > 0)
		{
			DataType* device_data;
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

	__global__ void GlobalCastRayFromCamera(int iteration, PerspectiveCamera* camera, PathSegment* pathSegment) 
	{
		int x = (blockIdx.x * blockDim.x) + threadIdx.x;
		int y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x >= camera->width || y >= camera->height) {
			return;
		}

		int index = (y * camera->width) + x;

		PathSegment& segment = pathSegment[index];

		segment.Reset();

		CudaRNG rng(iteration, index, 1);
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
		segment.intersection.Reset();

		scene.SceneIntersection(segment.ray, segment.intersection);
	}

	__global__ void GlobalDisplayNormal(int max_index, PathSegment* pathSegment, Scene scene)
	{
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index >= max_index || pathSegment[index].IsEnd())
		{
			return;
		}

		PathSegment& segment = pathSegment[index];
		Intersection& intersection = segment.intersection;
		if (intersection.id >= 0)
		{
			if (intersection.isLight)
			{
				segment.radiance = Spectrum(1.f);
			}
			else
			{
				Material* material = scene.materials[intersection.material_id];
				segment.surfaceNormal = material->GetNormal(intersection.normal, intersection.uv);
				//glm::vec3 wo = WorldToLocal(segment.surfaceNormal) * (-segment.ray.DIR);
				segment.radiance = 0.5f * (segment.surfaceNormal + 1.f);
			}
		}
		segment.End();
	}

	__global__ void GlobalNaiveLi(int iteration, int max_index, PathSegment* pathSegment, Scene scene)
	{
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index >= max_index || pathSegment[index].IsEnd())
		{
			return;
		}

		PathSegment& segment = pathSegment[index];
		Intersection& intersection = segment.intersection;
		Ray& ray = segment.ray;
		if (intersection.id >= 0)
		{
			if (intersection.isLight)
			{
				// hit light source
				segment.throughput *= scene.lights[intersection.id]->GetLe();
				segment.radiance += segment.throughput;
			}
			else
			{
				Material* material = scene.materials[intersection.material_id];
				segment.materialType = material->m_MaterialData.type;
				CudaRNG rng(iteration, index, 4 + segment.depth * 7);

				BSDF& bsdf = material->GetBSDF();

				const glm::vec3& normal = material->GetNormal(intersection.normal);

				BSDFSample bsdf_sample = bsdf.Sample_f(material->GetAlbedo(), segment.eta, -ray.DIR, normal, { rng.rand(), rng.rand() });

				if (bsdf_sample.pdf == 0.f && glm::length(bsdf_sample.f) == 0.f)
				{
					segment.End();
				}
				else
				{
					segment.bsdfPdf = bsdf_sample.pdf;
					segment.throughput *= bsdf_sample.f * AbsDot(bsdf_sample.wiW, normal) / bsdf_sample.pdf;
					segment.ray = Ray::SpawnRay(ray * intersection.t, bsdf_sample.wiW);
					++segment.depth;
					return;
				}
			}
		}
		segment.End();
	}
	
	__global__ void GlobalDirectLi(int iteration, int max_index, PathSegment* pathSegment, Scene scene)
	{
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index >= max_index || pathSegment[index].IsEnd())
		{
			return;
		}

		PathSegment& segment = pathSegment[index];
		Intersection& intersection = segment.intersection;
		Ray& ray = segment.ray;

		if(intersection.id >= 0.f)
		{
			if (intersection.isLight)
			{
				// hit light source
				segment.throughput *= scene.lights[intersection.id]->GetLe();
				segment.radiance += segment.throughput;
			}
			else
			{
				Material* material = scene.materials[intersection.material_id];
				glm::vec3 normal = intersection.normal;
				normal = material->GetNormal(normal);

				CudaRNG rng(iteration, index, 4 + segment.depth * 7);
				LightSample sample;

				if (scene.Sample_Li(rng.rand(), { rng.rand(), rng.rand() }, ray * intersection.t, normal, sample))
				{
					segment.throughput *= material->GetAlbedo() * sample.Le * AbsDot(sample.wiW, normal) / sample.pdf;
					
					segment.radiance += segment.throughput;
				}
			}
		}
		segment.End();
	}

	__global__ void GlobalMIS_Li(int iteration, int max_index, PathSegment* pathSegment, Scene scene)
	{
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index >= max_index || pathSegment[index].IsEnd())
		{
			return;
		}

		PathSegment& segment = pathSegment[index];
		Intersection& intersection = segment.intersection;
		Ray& ray = segment.ray;

		if (intersection.id >= 0)
		{
			if (intersection.isLight)
			{
				segment.throughput *= scene.lights[intersection.id]->GetLe();

				if (segment.depth > 0 && !MaterialIs(segment.materialType, MaterialType::Specular))
				{
					float light_pdf = scene.PDF_Li(intersection.id, ray * intersection.t, ray.DIR, intersection.t, segment.surfaceNormal);
					segment.throughput *= CudaPBRT::PowerHeuristic(1, segment.bsdfPdf, 1, light_pdf);
				}
				segment.radiance += segment.throughput;
			}
			else
			{
				Material* material = scene.materials[intersection.material_id];
				Spectrum albedo = material->GetAlbedo(intersection.uv);
				BSDF& bsdf = material->GetBSDF();
				segment.materialType = material->m_MaterialData.type;

				segment.surfaceNormal = material->GetNormal(intersection.normal, intersection.uv);
				const glm::vec3 surface_point = ray * intersection.t;

				const glm::vec3& normal = segment.surfaceNormal;
				CudaRNG rng(iteration, index, 4 + segment.depth * 7);

				// estimate direct light sample
				if (!MaterialIs(segment.materialType, MaterialType::Specular))
				{
					LightSample light_sample;
					if (scene.Sample_Li(rng.rand(), { rng.rand(), rng.rand() }, surface_point, normal, light_sample))
					{
						Spectrum scattering_f = bsdf.f(albedo, -ray.DIR, light_sample.wiW, normal); // evaluate scattering bsdf
						float scattering_pdf = bsdf.PDF(-ray.DIR, light_sample.wiW, normal);// evaluate scattering pdf
						if (scattering_pdf > 0.f)
						{
							segment.radiance += light_sample.Le * scattering_f * segment.throughput *
								CudaPBRT::PowerHeuristic(1, light_sample.pdf, 1, scattering_pdf) * AbsDot(light_sample.wiW, normal) / light_sample.pdf;
						}
					}
				}

				// compute throughput
				BSDFSample bsdf_sample = bsdf.Sample_f(albedo, segment.eta, -ray.DIR, normal, { rng.rand(), rng.rand() });

				if (bsdf_sample.pdf > 0.f)
				{
					segment.bsdfPdf = bsdf_sample.pdf;
					segment.throughput *= bsdf_sample.f * AbsDot(bsdf_sample.wiW, normal) / segment.bsdfPdf;
					segment.ray = Ray::SpawnRay(surface_point, bsdf_sample.wiW);
					++segment.depth;
					return;
				}
			}
		}
		
		segment.End();
	}

	__global__ void GlobalWritePixel(int iteration, int max_index, PathSegment* pathSegment, uchar4* img, float3* hdr_img)
	{
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		
		if (index >= max_index)
		{
			return;
		}
		PathSegment& segment = pathSegment[index];
		int& pixelId = segment.pixelId;

		writePixel(iteration, hdr_img[pixelId], img[pixelId], segment.radiance);
	}

	__global__ void GlobalDisplayTexture(CudaTexObj tex, int width, int height, float3* hdr_img, uchar4* img)
	{
		int x = (blockIdx.x * blockDim.x) + threadIdx.x;
		int y = (blockIdx.y * blockDim.y) + threadIdx.y;
		if (x >= width || y >= height) {
			return;
		}

		float u = static_cast<float>(x) / static_cast<float>(width);
		float v = static_cast<float>(y) / static_cast<float>(height);

		float4 albedo = ReadTexture(tex, { u, v });
		
		glm::vec3 color(albedo.x, albedo.y, albedo.z);
		color = color / (1.f + color);
		color = glm::pow(color, glm::vec3(1.f / 2.2f));
		color = glm::mix(glm::vec3(0.f), glm::vec3(255.f), color);
		int index = x + y * width;
		
		img[index].x = static_cast<int>(color.r);
		img[index].y = static_cast<int>(color.g);
		img[index].z = static_cast<int>(color.b);
		img[index].w = 255;
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

	void CudaPathTracer::DisplayTexture(const CudaTexture& texture)
	{
		auto dim = texture.GetDim();

		KernalConfig config({ width, height, 1 }, { 3, 3, 0 });
		GlobalDisplayTexture << < config.numBlocks, config.threadPerBlock >> > (texture.GetTextureObject(), width, height, device_hdr_image, device_image);
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

	void CudaPathTracer::Run(Scene* scene)
	{
		int max_count = width * height;
		
		auto devTerminatedThr = devTerminatedPathsThr;
		
		// cast ray from camera
		KernalConfig CamConfig({ width, height, 1 }, { 3, 3, 0 });
		GlobalCastRayFromCamera << < CamConfig.numBlocks, CamConfig.threadPerBlock >> > (m_Iteration, device_camera, device_pathSegment);
		cudaDeviceSynchronize();
		CUDA_CHECK_ERROR();

		int depth = 0;
		while (max_count > 0 && depth++ < CudaPBRT::PathMaxDepth)
		{
			KernalConfig intersectionConfig({ max_count, 1, 1 }, { 8, 0, 0 });

			// intersection
			GlobalSceneIntersection << < intersectionConfig.numBlocks, intersectionConfig.threadPerBlock >> > (max_count, device_pathSegment, *scene);
			cudaDeviceSynchronize();
			CUDA_CHECK_ERROR();

			//thrust::sort(devPathsThr, devPathsThr + max_count);

			KernalConfig throughputConfig({ max_count, 1, 1 }, { 8, 0, 0 });
			
			//GlobalDisplayNormal << < throughputConfig.numBlocks, throughputConfig.threadPerBlock >> > (max_count, device_pathSegment, *scene);

			//GlobalNaiveLi << <throughputConfig.numBlocks, throughputConfig.threadPerBlock >> > (m_Iteration, max_count, device_pathSegment, *scene);
			//GlobalDirectLi << <throughputConfig.numBlocks, throughputConfig.threadPerBlock >> > (m_Iteration, max_count, device_pathSegment, *scene);
			GlobalMIS_Li << <throughputConfig.numBlocks, throughputConfig.threadPerBlock >> > (m_Iteration, max_count, device_pathSegment, *scene);

			cudaDeviceSynchronize();
			CUDA_CHECK_ERROR();


			devTerminatedThr = thrust::remove_copy_if(devPathsThr, devPathsThr + max_count, devTerminatedThr, CompactTerminatedPaths());
			auto end = thrust::remove_if(devPathsThr, devPathsThr + max_count, RemoveInvalidPaths());
			max_count = end - devPathsThr;
		}
		int numContributing = devTerminatedThr.get() - device_terminatedPathSegment;
		KernalConfig pixelConfig({ numContributing, 1, 1 }, { 8, 0, 0 });
		
		GlobalWritePixel << <pixelConfig.numBlocks, pixelConfig.threadPerBlock >> > (m_Iteration, numContributing, device_terminatedPathSegment,
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
		
		++m_Iteration;
	}

	void CudaPathTracer::UpdateCamera(PerspectiveCamera& camera)
	{
		// Copy input vectors from host memory to GPU buffers.
		cudaMemcpy(device_camera, &camera, sizeof(PerspectiveCamera), cudaMemcpyHostToDevice);
		CUDA_CHECK_ERROR();
	}
}
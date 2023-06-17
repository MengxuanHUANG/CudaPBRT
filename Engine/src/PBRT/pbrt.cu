#include "pbrt.h"

#include "spectrum.h"
#include "reservior.h"

#include "pathSegment.h"
#include "gpuScene.h"

#include "texture.h"

#include "BVH/boundingBox.h"
#include "intersection.h"
#include "Shape/sphere.h"
#include "Shape/cube.h"
#include "Shape/triangle.h"

#include "Material/bsdf.h"
#include "Sampler/rng.h"
#include "Light/light.h"

#include <GL/glew.h>

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

	CPU_GPU void Create(Shape* shape_ptr, const ShapeData& data)
	{
		switch (data.type)
		{
		case ShapeType::Sphere:
			new (shape_ptr) Sphere(data);
			break;
		case ShapeType::Cube:
			new (shape_ptr) Cube(data);
			break;
		case ShapeType::Triangle:
			new (shape_ptr) Triangle(data);
			break;
		default:
			printf("Unknown ShapeType!\n");
		}
	}

	CPU_GPU void Create(Material* material_ptr, const MaterialData& data)
	{
		switch (data.type)
		{
		case MaterialType::LambertianReflection:
			new (material_ptr) Material(data);
			new (&material_ptr->m_BSDF) SingleBSDF();
			new (&(material_ptr->m_BSDF.m_GeneralData.bxdfs[0])) LambertianReflection();
			break;
		case MaterialType::SpecularReflection:
			new (material_ptr) Material(data);
			new (&material_ptr->m_BSDF) SingleBSDF();
			new (&(material_ptr->m_BSDF.m_GeneralData.bxdfs[0])) SpecularReflection();
			break;
		case MaterialType::SpecularTransmission:
			new (material_ptr) Material(data);
			new (&material_ptr->m_BSDF) SingleBSDF();
			new (&(material_ptr->m_BSDF.m_GeneralData.bxdfs[0])) SpecularTransmission(data.eta);
			break;
		case MaterialType::Glass:
			new (material_ptr) Material(data);
			new (&material_ptr->m_BSDF) FresnelBSDF();
			new (&(material_ptr->m_BSDF.m_GeneralData.bxdfs[0])) SpecularReflection();
			new (&(material_ptr->m_BSDF.m_GeneralData.bxdfs[1])) SpecularTransmission(data.eta);
			break;
		case MaterialType::MicrofacetReflection:
			new (material_ptr) Material(data);
			new (&material_ptr->m_BSDF) SingleBSDF();
			new (&(material_ptr->m_BSDF.m_GeneralData.bxdfs[0])) MicrofacetReflection();
			break;
		case MaterialType::MetallicWorkflow:
			new (material_ptr) Material(data);
			new (&material_ptr->m_BSDF) SingleBSDF();
			new (&(material_ptr->m_BSDF.m_GeneralData.bxdfs[0])) MetallicWorkflow();
			break;
		default:
			printf("Unknown MaterialType!\n");
		}
	}

	CPU_GPU void Create(Light* light_ptr, const LightData& data)
	{
		switch (data.type)
		{
		case LightType::ShapeLight:
			new (light_ptr) ShapeLight(data);
			break;
		default:
			printf("Unknown LightType!\n");
		}
	}

	CPU_GPU Ray CastRay(const PerspectiveCamera& camera, const glm::vec2& p, RNG& rng)
	{
		glm::vec2 ndc = 2.f * (p + glm::vec2(rng.rand(), rng.rand())) / glm::vec2(camera.width, camera.height);
		ndc.x = ndc.x - 1.f;
		ndc.y = 1.f - ndc.y;

		float aspect = static_cast<float>(camera.width) / static_cast<float>(camera.height);

		// point in camera space
		float radian = glm::radians(camera.fovy * 0.5f);
		glm::vec3 p_camera = glm::vec3(
			ndc.x * glm::tan(radian) * aspect,
			ndc.y * glm::tan(radian),
			1.f
		);

		Ray ray(glm::vec3(0), p_camera);

		if (camera.lensRadius > 0.f)
		{
			glm::vec2 p_lens(rng.rand(), rng.rand());
			p_lens = camera.lensRadius * Sampler::SquareToDiskConcentric(p_lens);
			glm::vec3 p_focus = camera.focalDistance * p_camera;

			ray.O.x = p_lens.x;
			ray.O.y = p_lens.y;
			ray.DIR = glm::normalize(p_focus - ray.O);
		}

		// transform to world space
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

		color = glm::mix(preColor, color, 1.f / float(iteration + 1));
		
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
	__global__ void CreateArray(T* device_array, DataType* data, size_t max_count)
	{
		int id = blockIdx.x;
		if (id >= max_count)
		{
			return;
		}
		Create(device_array + id, data[id]);
	}

	template<typename T>
	__global__ void FreeArray(T** device_array, size_t max_count)
	{
		for (int i = 0; i < max_count; ++i)
		{
			SAFE_FREE(device_array[i]);
		}
	}

	template<typename T, typename DataType>
	void CreateArrayOnCuda<T, DataType>(T*& dev_array, size_t& count, std::vector<DataType>& host_data)
	{
		count = host_data.size();
		if (count > 0 && dev_array == nullptr)
		{
			DataType* device_data;
			cudaMalloc((void**)&device_data, sizeof(DataType) * count);
			CUDA_CHECK_ERROR();

			cudaMemcpy(device_data, host_data.data(), sizeof(DataType) * count, cudaMemcpyHostToDevice);
			CUDA_CHECK_ERROR();

			cudaMalloc((void**)&dev_array, sizeof(T) * count);
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

	template void CreateArrayOnCuda<Light, LightData>(Light*& dev_array, size_t& dev_count, std::vector<LightData>& host_data);
	template void CreateArrayOnCuda<Material, MaterialData>(Material*& dev_array, size_t& dev_count, std::vector<MaterialData>& host_data);
	template void CreateArrayOnCuda<Shape, ShapeData>(Shape*& dev_array, size_t& dev_count, std::vector<ShapeData>& host_data);

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

	template<typename T, typename DataType>
	void UpdateArrayOnCuda(T*& dev_array, std::vector<DataType>& host_data, size_t start, size_t end)
	{
		size_t count = end - start;
		if (start >= 0 && end <= host_data.size() && count > 0)
		{
			
			DataType* device_data;
			cudaMalloc((void**)&device_data, sizeof(DataType) * count);
			CUDA_CHECK_ERROR();

			cudaMemcpy(device_data, host_data.data() + start, sizeof(DataType) * count, cudaMemcpyHostToDevice);
			CUDA_CHECK_ERROR();

			KernalConfig createConfig({ count, 1, 1 }, { 0, 0, 0 });
			CreateArray<T, DataType> << < createConfig.numBlocks, createConfig.threadPerBlock >> > (dev_array + start, device_data, count);

			// cudaDeviceSynchronize waits for the kernel to finish
			cudaDeviceSynchronize();
			CUDA_CHECK_ERROR();

			CUDA_FREE(device_data);
		}
	}

	template void UpdateArrayOnCuda<Light, LightData>(Light*& dev_array, std::vector<LightData>& data, size_t start, size_t end);
	template void UpdateArrayOnCuda<Shape, ShapeData>(Shape*& dev_array, std::vector<ShapeData>& data, size_t start, size_t end);
	template void UpdateArrayOnCuda<Material, MaterialData>(Material*& dev_array, std::vector<MaterialData>& data, size_t start, size_t end);

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
		segment.ray = CastRay(*camera, { x, y }, rng);
		segment.pixelId = index;
	}

	__global__ void GlobalSceneIntersection(int max_index, PathSegment* pathSegment, GPUScene scene)
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

	__global__ void GlobalDisplayNormal(int max_index, PathSegment* pathSegment, GPUScene scene)
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
			Material* material = scene.materials + intersection.material_id;
			float Lv = material->GetLv(intersection.uv);
			if (Lv > 0.f)
			{
				glm::vec2 uv = scene.shapes[intersection.id].GetUV(intersection.p);
				segment.surfaceNormal = material->GetNormal(scene.shapes[intersection.id].GetNormal(intersection.p), uv);
			}
			else
			{
				segment.surfaceNormal = material->GetNormal(intersection.normal, intersection.uv);
				//glm::vec3 wo = WorldToLocal(segment.surfaceNormal) * (-segment.ray.DIR);
			}
			segment.radiance = 0.5f * (segment.surfaceNormal + 1.f);
		}
		segment.End();
	}

	__global__ void GlobalNaiveLi(int iteration, int max_index, PathSegment* pathSegment, GPUScene scene)
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
			Material* material = scene.materials + intersection.material_id;
			float Lv = material->GetLv(intersection.uv);

			if (Lv > 0.f)
			{
				// hit light source
				segment.throughput *= material->GetIrradiance(intersection.uv);
				segment.radiance += segment.throughput;
			}
			else
			{
				Spectrum albedo = material->GetAlbedo(intersection.uv);
				segment.materialType = material->m_MaterialData.type;

				const glm::vec3& normal = material->GetNormal(intersection.normal, intersection.uv);
				float roughness = material->GetRoughness(intersection.uv);
				float metallic = material->GetMetallic(intersection.uv);
				BSDFData bsdf_data(normal, roughness, metallic, segment.eta, albedo);

				CudaRNG rng(iteration, index, 4 + segment.depth * 7);
				const BSDF& bsdf = material->GetBSDF();
				BSDFSample bsdf_sample = bsdf.Sample_f(bsdf_data, -ray.DIR, rng);

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
	
	__global__ void GlobalDirectLi(int iteration, int max_index, PathSegment* pathSegment, GPUScene scene)
	{
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index >= max_index || pathSegment[index].IsEnd())
		{
			return;
		}

		PathSegment& segment = pathSegment[index];
		const Intersection& intersection = segment.intersection;
		const Ray& ray = segment.ray;

		if(intersection.id >= 0)
		{
			const Material& material = scene.materials[intersection.material_id];
			float Lv = material.GetLv(intersection.uv);

			if (Lv > 0.f)
			{
				// hit light source
				segment.throughput *= material.GetIrradiance(intersection.uv);
				segment.radiance += segment.throughput;
			}
			else
			{
				Spectrum albedo = material.GetAlbedo(intersection.uv);
				const BSDF& bsdf = material.GetBSDF();
				segment.materialType = material.m_MaterialData.type;

				segment.surfaceNormal = material.GetNormal(intersection.normal, intersection.uv);
				float roughness = material.GetRoughness(intersection.uv);
				float metallic = material.GetMetallic(intersection.uv);

				const glm::vec3& normal = segment.surfaceNormal;

				glm::mat3 world_to_local = WorldToLocal(normal);
				glm::vec3 wo = glm::normalize(world_to_local * -ray.DIR);

				BSDFData bsdf_data(normal, roughness, metallic, segment.eta, albedo);

				CudaRNG rng(iteration, index, 4 + segment.depth * 7);
				if (scene.light_count > 0 && !MaterialIs(segment.materialType, MaterialType::Specular))
				{
					Reservior<LightSample> temporal_reservior;

					// obtain previous reservior
					Reservior<LightSample> pre_reservior;

					if (iteration > 0)
					{
						// TODO: compute motion vector to locate current pixel in the buffer
						int pre_index = segment.pixelId; // currently, assume there is no shifting

						pre_reservior = scene.temporalReserviors[pre_index];
					}

					// standard RIS DI
					Reservior<LightSample> current_reservior;

					for (int i = 0; i < scene.M; ++i)
					{
						LightSample light_sample;
						if (scene.Sample_Li(rng, intersection.p, normal, light_sample))
						{
							glm::vec3 wi = glm::normalize(world_to_local * light_sample.wiW);

							Spectrum scattering_f = bsdf.f(bsdf_data, wo, wi) * AbsDot(light_sample.wiW, normal);

							current_reservior.Update(rng.rand(), light_sample, glm::length(scattering_f) / light_sample.pdf);
						}
					}
					
					const LightSample& light_sample = current_reservior.y;
					float t = glm::length(light_sample.p - intersection.p);
					float p_hat = 0.f;
					if (current_reservior.M > 0)
					{
						if (!scene.Occluded(t, light_sample.light->GetShapeId(), Ray::SpawnRay(intersection.p, light_sample.wiW)))
						{
							glm::vec3 wi = glm::normalize(world_to_local * light_sample.wiW);

							Spectrum scattering_f = bsdf.f(bsdf_data, wo, wi) * AbsDot(light_sample.wiW, normal);

							p_hat = glm::length(scattering_f);
							
							current_reservior.weightSum = glm::max(0.0001f, current_reservior.weightSum);

							current_reservior.W = (current_reservior.weightSum / (p_hat * current_reservior.M));
						}
						else
						{
							current_reservior.W = 0.f;
						}
						temporal_reservior.Merge(rng.rand(), current_reservior, p_hat);
					}

					// temporal reuse
					if (scene.temporalReuse && pre_reservior.M > 0)
					{
						LightSample& pre_sample = pre_reservior.y;
						glm::vec3 wi = glm::normalize(world_to_local * pre_sample.wiW);
						Spectrum scattering_f = bsdf.f(bsdf_data, wo, wi) * AbsDot(pre_sample.wiW, normal);
					
						float p_hat = glm::length(scattering_f);
					
						if (pre_reservior.M > 20 * current_reservior.M)
						{
							pre_reservior.weightSum *= static_cast<float>(20 * current_reservior.M) / static_cast<float>(pre_reservior.M);
							pre_reservior.M = 20 * current_reservior.M;
						}

						pre_reservior.Update(rng.rand(), temporal_reservior.y, temporal_reservior.weightSum);
						temporal_reservior = pre_reservior;
					}
					

					// TODO: spatio reuse
					Reservior<LightSample> spatio_reservior;

					if (temporal_reservior.M > 0)
					{
						const LightSample& final_sample = temporal_reservior.y;

						glm::vec3 wi = glm::normalize(world_to_local * final_sample.wiW);
						Spectrum scattering_f = bsdf.f(bsdf_data, wo, wi) * AbsDot(final_sample.wiW, normal);
						p_hat = glm::length(scattering_f);
						
						temporal_reservior.W = temporal_reservior.weightSum / (p_hat * temporal_reservior.M);
						
						scene.temporalReserviors[segment.pixelId] = temporal_reservior;

						if (temporal_reservior.W > 0.f)
						{
							segment.throughput *= scattering_f * final_sample.light->GetLe(final_sample.p) * temporal_reservior.W;
							segment.radiance += segment.throughput;
						}
					}
				}
			}
		}
		segment.End();
	}

	__global__ void GlobalMIS_Li(int iteration, int max_index, PathSegment* pathSegment, GPUScene scene)
	{
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index >= max_index || pathSegment[index].IsEnd())
		{
			return;
		}

		PathSegment& segment = pathSegment[index];
		Intersection& intersection = segment.intersection;
		Ray& ray = segment.ray;
		const EnvironmentMap& env_map = scene.envMap;
		if (intersection.id >= 0)
		{
			const Material& material = scene.materials[intersection.material_id];

			if ((material.m_MaterialData.lightMaterial) || material.GetLv(intersection.uv) > 0.f)
			{
				segment.throughput *= material.GetIrradiance(intersection.uv);

				if (material.m_MaterialData.lightMaterial 
					&& segment.depth > 0 
					&& !MaterialIs(segment.materialType, MaterialType::Specular))
				{
					float light_pdf = scene.PDF_Li(intersection.id, ray * intersection.t, ray.DIR, intersection.t, segment.surfaceNormal);
					segment.throughput *= CudaPBRT::PowerHeuristic(1, segment.bsdfPdf, 1, light_pdf);
				}
				segment.radiance += segment.throughput;
			}
			else
			{
				Spectrum albedo = material.GetAlbedo(intersection.uv);
				const BSDF& bsdf = material.GetBSDF();
				segment.materialType = material.m_MaterialData.type;

				segment.surfaceNormal = material.GetNormal(intersection.normal, intersection.uv);
				float roughness = material.GetRoughness(intersection.uv);
				float metallic = material.GetMetallic(intersection.uv);

				const glm::vec3& normal = segment.surfaceNormal;

				glm::mat3 world_to_local = WorldToLocal(normal);
				glm::vec3 wo = glm::normalize(world_to_local * -ray.DIR);

				BSDFData bsdf_data(normal, roughness, metallic, segment.eta, albedo);

				CudaRNG rng(iteration, index, 4 + segment.depth * 7);

				// estimate direct light sample
				if (scene.light_count > 0 && !MaterialIs(segment.materialType, MaterialType::Specular))
				{
					Reservior<LightSample> light_sample_reservior;

					for (int i = 0; i < scene.M; ++i)
					{
						LightSample light_sample;
						if (scene.Sample_Li(rng, intersection.p, normal, light_sample))
						{
							glm::vec3 wi = glm::normalize(world_to_local * light_sample.wiW);

							Spectrum scattering_f = bsdf.f(bsdf_data, wo, wi) * AbsDot(light_sample.wiW, normal);
							float scattering_pdf = bsdf.PDF(bsdf_data, wo, wi);
							if (scattering_pdf > 0.01f)
							{
								light_sample_reservior.Update(rng.rand(), light_sample, CudaPBRT::PowerHeuristic(1, light_sample.pdf, 1, scattering_pdf) * glm::length(scattering_f) / light_sample.pdf);
							}
						}
					}

					const LightSample& light_sample = light_sample_reservior.y;
					float t = glm::length(light_sample.p - intersection.p);

					if (light_sample_reservior.M > 0 && !scene.Occluded(t, light_sample.light->GetShapeId(), Ray::SpawnRay(intersection.p, light_sample.wiW)))
					{
						glm::vec3 wi = glm::normalize(world_to_local * light_sample.wiW);

						Spectrum scattering_f = bsdf.f(bsdf_data, wo, wi) * AbsDot(light_sample.wiW, normal);

						light_sample_reservior.W = light_sample_reservior.weightSum / (light_sample_reservior.M * glm::length(scattering_f));
						if (light_sample_reservior.W > 0.01f)
						{
							segment.radiance += scattering_f * light_sample.light->GetLe(light_sample.p) * segment.throughput * light_sample_reservior.W;
						}
					}
				}

				// compute throughput
				BSDFSample bsdf_sample = bsdf.Sample_f(bsdf_data, wo, rng);

				if (bsdf_sample.pdf > 0.f)
				{
					segment.bsdfPdf = bsdf_sample.pdf;
					segment.throughput *= bsdf_sample.f * AbsDot(bsdf_sample.wiW, normal) / segment.bsdfPdf;
					segment.ray = Ray::SpawnRay(intersection.p, bsdf_sample.wiW);
					++segment.depth;
					return;
				}
			}
		}
		else if(env_map.GetTexObj() > 0)
		{
			float4 irradiance = env_map.GetIrradiance(ray.DIR);
			segment.throughput *= Spectrum(irradiance.x, irradiance.y, irradiance.z);
			segment.radiance += segment.throughput;
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
		
		float a = static_cast<float>(x) / static_cast<float>(width);

		hdr_img[index].x = glm::mix(0.f, 40.f, 1.f - a);
		hdr_img[index].y = glm::mix(0.f, 40.f, 1.f - a);
		hdr_img[index].z = glm::mix(0.f, 40.f, 1.f - a);

		img[index].x = 255;//static_cast<int>(color.r);
		img[index].y = 255;//static_cast<int>(color.g);
		img[index].z = 255;//static_cast<int>(color.b);
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

	void CudaPathTracer::Run(GPUScene* scene)
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
			GlobalDirectLi << <throughputConfig.numBlocks, throughputConfig.threadPerBlock >> > (m_Iteration, max_count, device_pathSegment, *scene);
			//GlobalMIS_Li << <throughputConfig.numBlocks, throughputConfig.threadPerBlock >> > (m_Iteration, max_count, device_pathSegment, *scene);

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
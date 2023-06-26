#pragma once

#include "pbrtDefine.h"
#include "Camera/Camera.h"
#include "ray.h"

#include <thrust/device_ptr.h>

#include <vector>

namespace CudaPBRT
{
	class LightSample;

	template<typename T>
	class Reservior;

	class GPUScene;
	class CudaTexture;
	class RNG;
	class Intersection;

	CPU_GPU Ray CastRay(const PerspectiveCamera& camera, const glm::vec2& p, RNG& rng);

	struct PathSegment;
	template<typename T>
	void BufferData(T*& device_ptr, const T* host_ptr, size_t size)
	{
		ASSERT(device_ptr == nullptr);
		if (size > 0)
		{
			ASSERT(host_ptr != nullptr);

			cudaMalloc((void**)&device_ptr, sizeof(T) * size);
			CUDA_CHECK_ERROR();

			cudaMemcpy(device_ptr, host_ptr, sizeof(T) * size, cudaMemcpyHostToDevice);
			CUDA_CHECK_ERROR();
		}
	}

	template<typename T, typename DataType>
	void CreateArrayOnCuda(T*& dev_array, size_t& dev_count, std::vector<DataType>& host_data);

	template<typename T>
	void FreeArrayOnCuda(T**& device_array, size_t count);

	template<typename T, typename DataType>
	void UpdateArrayOnCuda(T*& dev_array, std::vector<DataType>& host_data, size_t start, size_t end);

	struct GBuffer
	{
		Reservior<LightSample>* preReserviors = nullptr;
		Reservior<LightSample>* curReserviors = nullptr;
		
		Reservior<LightSample>* intermediaReserviors = nullptr;

		Intersection* preGeometryInfos = nullptr;
		Intersection* curGeometryInfos = nullptr;

		void Swap()
		{
			{
				Reservior<LightSample>* temp = preReserviors;
				preReserviors = curReserviors;
				curReserviors = temp;
			}
			{
				Intersection* temp = preGeometryInfos;
				preGeometryInfos = curGeometryInfos;
				curGeometryInfos = temp;
			}
		}
	};

	class CudaPathTracer
	{
	public:
		CudaPathTracer();
		virtual ~CudaPathTracer();

		virtual void InitCuda(PerspectiveCamera& camera, int device = 0);

		virtual void FreeCuda();
		void DisplayTexture(const CudaTexture& texture);
		virtual void Run(GPUScene* scene);

		virtual void UpdateCamera(PerspectiveCamera& camera);
		virtual unsigned int GetDisplayTextureId() const { return m_DisplayImage; }

		virtual void SwapGBuffer();

		inline void ResetPRBT() { m_Iteration = 0; }

	public:
		int m_Iteration; // number of iteration

		int width, height;

		// texture handler
		unsigned int m_DisplayImage = 0;

		GBuffer GBuffer; // 0 for last frame, 1 for current frame

		PerspectiveCamera* device_camera = nullptr;
		uchar4* device_image = nullptr;
		float3* device_hdr_image = nullptr;

		uchar4* host_image = nullptr;

		PathSegment* device_pathSegment = nullptr;
		PathSegment* device_terminatedPathSegment = nullptr;

		thrust::device_ptr<PathSegment> devPathsThr;
		thrust::device_ptr<PathSegment> devTerminatedPathsThr;
	};

	struct KernalConfig
	{
		dim3 numBlocks;
		dim3 threadPerBlock;

		KernalConfig(const glm::vec3& blocks, const glm::ivec3& threads)
		{
			threadPerBlock = dim3(BIT(threads.x), BIT(threads.y), BIT(threads.z));

			numBlocks = dim3(glm::ceil(blocks.x / static_cast<float>(threadPerBlock.x)),
							 glm::ceil(blocks.y / static_cast<float>(threadPerBlock.y)),
							 glm::ceil(blocks.z / static_cast<float>(threadPerBlock.z)));
		}
	};
}

#pragma once

#include "pbrtDefine.h"
#include "Camera/Camera.h"

#include <thrust/device_ptr.h>

#include <vector>

namespace CudaPBRT
{
	class Scene;

	struct PathSegment;
	template<typename T>
	void BufferData(T*& device_ptr, T* host_ptr, size_t size)
	{
		ASSERT(device_ptr == nullptr);
		ASSERT(host_ptr != nullptr);
		if (size > 0)
		{
			cudaMalloc((void**)&device_ptr, sizeof(T) * size);
			CUDA_CHECK_ERROR();

			cudaMemcpy(device_ptr, host_ptr, sizeof(T) * size, cudaMemcpyHostToDevice);
			CUDA_CHECK_ERROR();
		}
	}

	template<typename T, typename DataType>
	void CreateArrayOnCude(T**& dev_array, size_t& dev_count, std::vector<DataType>& host_data);

	template<typename T>
	void FreeArrayOnCuda(T**& device_array, size_t count);

	class CudaPathTracer
	{
	public:
		CudaPathTracer();
		virtual ~CudaPathTracer();

		virtual void InitCuda(PerspectiveCamera& camera, int device = 0);

		virtual void FreeCuda();
		virtual void Run(Scene* scene);

		virtual void UpdateCamera(PerspectiveCamera& camera);
		virtual unsigned int GetDisplayTextureId() const { return m_DisplayImage; }

		inline void ResetPRBT() { m_Iteration = 1; }

	public:
		int m_Iteration; // number of iteration

		int width, height;

		// texture handler
		unsigned int m_DisplayImage = 0;

		PerspectiveCamera* device_camera = nullptr;
		uchar4* device_image = nullptr;
		float3* device_hdr_image = nullptr;

		uchar4* host_image = nullptr;

		PathSegment* device_pathSegment = nullptr;
		PathSegment* device_terminatedPathSegment = nullptr;

		thrust::device_ptr<PathSegment> devPathsThr;
		thrust::device_ptr<PathSegment> devTerminatedPathsThr;
	};
}

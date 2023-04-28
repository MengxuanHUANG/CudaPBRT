#pragma once

#include "Core/Core.h"

#include <limits>
#include <cmath>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#define GPU_ONLY __device__
#define CPU_ONLY __host__
#define CPU_GPU __host__ __device__

#define INLINE __forceinline__

#define CUDA_FREE(ptr) if(ptr != nullptr) { cudaFree(ptr); ptr = nullptr; }

#define USE_BVH 1

#if USE_BVH
#define BVH_SAH 1
#endif

namespace CudaPBRT
{
	typedef cudaTextureObject_t CudaTexObj;
	typedef cudaArray_t CudaArray;

	static constexpr int PathMaxDepth = 5;

	static constexpr float ShadowEpsilon	= 0.0001f;
	static constexpr float Pi				= 3.1415927f;
	static constexpr float InvPi			= 0.3183099f;
	static constexpr float Inv2Pi			= 0.1591549f;
	static constexpr float Inv4Pi			= 0.0795775f;
	static constexpr float PiOver2			= 1.5707963f;
	static constexpr float PiOver4			= 0.7853981f;
	static constexpr float Sqrt2			= 1.4142136f;

	static constexpr float FloatMin = -10000000.f;
	static constexpr float FloatMax =  10000000.f;
	static constexpr float FloatEpsilon = std::numeric_limits<float>::epsilon();

	static constexpr float MachineEpsilon = 0.5f * std::numeric_limits<float>::epsilon();

	static constexpr float AirETA = 1.000293f;

	CPU_GPU inline float gamma(int n)
	{
		return n * MachineEpsilon / (1.f - n * MachineEpsilon);
	}

	inline int bit_length(int n)
	{
		return static_cast<int>(ceil(log2(n)));
	}

	inline int UpperBinary(int value)
	{
		return (1 << bit_length(value));
	}

	INLINE CPU_GPU float AbsDot(const glm::vec3& wi, const glm::vec3& nor) { return glm::abs(glm::dot(wi, nor)); }

	INLINE CPU_GPU float CosTheta(const glm::vec3& w) { return w.z; }
	INLINE CPU_GPU float AbsCosTheta(const glm::vec3& w) { return glm::abs(w.z); }
	INLINE CPU_GPU  float PowerHeuristic(int nf, float fPdf, int ng, float gPdf) 
	{
		float f = static_cast<float>(nf) * fPdf;
		float g = static_cast<float>(ng) * gPdf;

		return (f * f) / (f * f + g * g);
	}
	
	INLINE CPU_GPU void coordinateSystem(const glm::vec3& v1, glm::vec3& v2, glm::vec3& v3)
	{
		if (glm::abs(v1.x) > glm::abs(v1.y))
			v2 = glm::vec3(-v1.z, 0, v1.x) / glm::sqrt(v1.x * v1.x + v1.z * v1.z);
		else
			v2 = glm::vec3(0, v1.z, -v1.y) / glm::sqrt(v1.y * v1.y + v1.z * v1.z);
		v3 = glm::cross(v1, v2);
	}

	INLINE CPU_GPU glm::mat3 LocalToWorld(const::glm::vec3& nor)
	{
		glm::vec3 tan, bit;
		coordinateSystem(nor, tan, bit);
		return glm::mat3(tan, bit, nor);
	}
	INLINE CPU_GPU glm::mat3 WorldToLocal(const::glm::vec3& nor)
	{
		return glm::transpose(LocalToWorld(nor));
	}
	
	template<typename T>
	INLINE CPU_GPU T BarycentricInterpolation(const T& c1, const T& c2, const T& c3, float u, float v)
	{
		return u * c1 + v * c2 + (1.f - u - v) * c3;
	}

	inline void CudaCheckError(const char* file, int line)
	{
		cudaDeviceSynchronize();
		cudaError_t status = cudaGetLastError();
		if (status != cudaSuccess) 
		{
			fprintf(stderr, "(%s:%d): %s\n", file, line, cudaGetErrorString(status));
		}
	}

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

#ifdef CUDA_PBRT_DEBUG
#define CUDA_CHECK_ERROR() CudaPBRT::CudaCheckError(__FILE__, __LINE__)
#else 
#define CUDA_CHECK_ERROR()
#endif


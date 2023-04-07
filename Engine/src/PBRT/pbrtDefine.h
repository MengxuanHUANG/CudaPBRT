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

namespace CudaPBRT
{
	static constexpr int PathMaxDepth = 10;

	static constexpr float ShadowEpsilon	= 0.0001f;
	static constexpr float Pi				= 3.1415927f;
	static constexpr float InvPi			= 0.3183099f;
	static constexpr float Inv2Pi			= 0.1591549f;
	static constexpr float Inv4Pi			= 0.0795775f;
	static constexpr float PiOver2			= 1.5707963f;
	static constexpr float PiOver4			= 0.7853981f;
	static constexpr float Sqrt2			= 1.4142136f;

	static constexpr float FloatMin = std::numeric_limits<float>::min();
	static constexpr float FloatMax = std::numeric_limits<float>::max();
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

	INLINE CPU_GPU  float PowerHeuristic(int nf, float fPdf, int ng, float gPdf) 
	{
		float f = static_cast<float>(nf) * fPdf;
		float g = static_cast<float>(ng) * gPdf;

		return (f * f) / (f * f + g * g);
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


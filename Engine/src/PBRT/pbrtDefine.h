#pragma once

#include "Core/Core.h"

#include <limits>
#include <cmath>
#include <cuda_runtime.h>

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

	inline void CudaCheckError(const char* file, int line)
	{
		cudaDeviceSynchronize();
		cudaError_t status = cudaGetLastError();
		if (status != cudaSuccess) 
		{
			fprintf(stderr, "(%s:%d): %s\n", file, line, cudaGetErrorString(status));
		}
	}
}

#ifdef CUDA_PBRT_DEBUG
#define CUDA_CHECK_ERROR() CudaPBRT::CudaCheckError(__FILE__, __LINE__)
#else 
#define CUDA_CHECK_ERROR()
#endif


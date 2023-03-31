#pragma once

#include <limits>

#include <cuda_runtime.h>

#ifndef PBRT_DEFINES
#define PBRT_DEFINES

#define GPU_ONLY __device__
#define CPU_ONLY __host__
#define CPU_GPU __host__ __device__
#endif

namespace CudaPBRT
{
	GPU_ONLY static float ShadowEpsilon = 0.0001f;
	GPU_ONLY static float Pi = 3.14159265358979323846;
	GPU_ONLY static float InvPi = 0.31830988618379067154;
	GPU_ONLY static float Inv2Pi = 0.15915494309189533577;
	GPU_ONLY static float Inv4Pi = 0.07957747154594766788;
	GPU_ONLY static float PiOver2 = 1.57079632679489661923;
	GPU_ONLY static float PiOver4 = 0.78539816339744830961;
	GPU_ONLY static float Sqrt2 = 1.41421356237309504880;

	GPU_ONLY static float FloatMin = std::numeric_limits<float>::min();
	GPU_ONLY static float FloatMax = std::numeric_limits<float>::max();
	GPU_ONLY static float MachineEpsilon = 0.5f * std::numeric_limits<float>::epsilon();

	GPU_ONLY inline float gamma(int n)
	{
		return n * MachineEpsilon / (1.f - n * MachineEpsilon);
	}
}


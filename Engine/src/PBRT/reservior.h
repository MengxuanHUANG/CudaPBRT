#pragma once

#include "pbrtDefine.h"

namespace CudaPBRT
{
	template<typename T>
	class Reservior
	{
	public:
		CPU_GPU Reservior()
			: weightSum(0.f), M(0), W(0.f)
		{}

		CPU_GPU void Update(float rand_num, const T& xi, float weight)
		{
			weightSum += weight;
			M += 1;
			if (rand_num < (weight / weightSum)) y = xi;
		}

	public:
		T y;
		float weightSum;
		int M;
		float W;
	};
}
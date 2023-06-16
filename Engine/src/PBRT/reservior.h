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

		INLINE CPU_GPU void Update(float rand_num, const T& xi, float weight)
		{
			weightSum += weight;
			M += 1;
			if (rand_num <= (weight / weightSum)) y = xi;
		}
		INLINE CPU_GPU void Merge(float rand_num, const Reservior<T>& other, float target_pdf)
		{
			int M0 = M;
			Update(rand_num, other.y, target_pdf * other.W * other.M);
			M = M0 + other.M;
		}

	public:
		T y;
		float weightSum;
		int M;
		float W;
	};
}
#pragma once

#include "pbrtDefine.h"

namespace CudaPBRT
{
	template<typename T>
	class Reservior
	{
	public:
		CPU_GPU Reservior()
			: weightSum(0.f), M(0), W(0.f), y()
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

		CPU_GPU Reservior<T> operator=(const Reservior<T>& other)
		{
			y = other.y;
			weightSum = other.weightSum;
			M = other.M;
			W = other.W;
			return *this;
		}

		INLINE CPU_GPU void Reset()
		{
			this->y = T();
			this->weightSum = 0.f;
			this->M = 0;
			this->W = 0.f;
		}

	public:
		T y;
		float weightSum;
		int M;
		float W;
	};
}
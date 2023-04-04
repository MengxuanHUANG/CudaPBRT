#pragma once
#include "PBRT/pbrtDefine.h"

#include <thrust/random.h>

namespace CudaPBRT
{
	class RNG
	{
	public:
		CPU_GPU virtual float rand() = 0;
	};

	class CudaRNG : public RNG
	{
	public:
		CPU_GPU CudaRNG(int iter, int index, int dim)
		{
			int h = utilhash((1 << 31) | (dim << 22) | iter) ^ utilhash(index);
			rng = thrust::default_random_engine(h);
		}

		CPU_GPU virtual float rand() override
		{
			return thrust::uniform_real_distribution<float>(0.f, 1.f)(rng);
		}
	protected:
		// Handy-dandy hash function that provides seeds for random number generation.
		INLINE CPU_GPU unsigned int utilhash(unsigned int a) 
		{
			a = (a + 0x7ed55d16) + (a << 12);
			a = (a ^ 0xc761c23c) ^ (a >> 19);
			a = (a + 0x165667b1) + (a << 5);
			a = (a + 0xd3a2646c) ^ (a << 9);
			a = (a + 0xfd7046c5) + (a << 3);
			a = (a ^ 0xb55a4f09) ^ (a >> 16);
			return a;
		}
	
	protected:
		thrust::default_random_engine rng;
	};

	class HashRNG : public RNG
	{
	public:
		CPU_GPU HashRNG(const glm::vec2& seed)
			: m_Seed(seed)
		{}

		CPU_GPU virtual float rand() override
		{
			return hash(m_Seed++);
		}

	protected:
		// from https://www.shadertoy.com/view/fsKBzw
		CPU_GPU float hash(glm::vec2 f)
		{
			glm::uvec2 x = glm::uvec2(glm::floatBitsToUint(f.x), glm::floatBitsToUint(f.y));
			glm::uvec2 q = 1103515245U * (x >> 1U ^ glm::uvec2(x.y, x.x));

			return static_cast<float>(1103515245U * (q.x ^ q.y >> 3U)) / static_cast<float>(0xffffffffU);
		}

	protected:
		glm::vec2 m_Seed;
	};
}
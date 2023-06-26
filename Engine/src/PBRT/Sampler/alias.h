#pragma once

#include "PBRT/pbrtDefine.h"

#include <queue>

namespace CudaPBRT
{
	class Cuda_AlaisSampler_1D
	{
	public:
		CPU_ONLY Cuda_AlaisSampler_1D(int N, const float* distribution);
		CPU_ONLY ~Cuda_AlaisSampler_1D();

	protected:
		CPU_ONLY void CreateSampler(int N, const float* distribution);
		CPU_ONLY void FreeSampler();

	public:
		float* device_distribution;
		float* device_accept;
		int* device_alias;
		int N;
	};

	class AlaisSampler_1D
	{
	public:
		CPU_GPU AlaisSampler_1D(int N, const float* distribution, float* accept, int* alias)
			:N(N), accept(accept), alias(alias)
		{
		}

		INLINE CPU_GPU int Sample(const glm::vec2& xi)
		{
			return Sample(xi, N, accept, alias);
		}


		INLINE CPU_GPU static int Sample(const glm::vec2& xi, int N, float* accept, int* alias)
		{
			int i = (N - 1) * xi.x;
			return (xi.y <= accept[i] ? i : alias[i]);
		}

		INLINE CPU_ONLY static void Create(int N, const float* distribution, float* accept, int* alias)
		{
			struct pair
			{
				pair(int id, float value)
					:id(id), value(value)
				{}

				int id;
				float value;
				inline bool operator<(const pair& other) const { return value < other.value; }
			};

			std::priority_queue<pair> greater, smaller;
			
			for (int i = 0; i < N; ++i)
			{
				accept[i] = 0.f;
				alias[i] = -1;

				if (distribution[i] == 1.f)
				{
					accept[i] = 1.f;
					alias[i] = -1;
				}
				else if (distribution[i] > 1.f)
				{
					greater.emplace(i, distribution[i]);
				}
				else
				{
					smaller.emplace(i, distribution[i]);
				}
			}

			while (!smaller.empty() && !greater.empty())
			{
				auto p_greater = greater.top();
				greater.pop();
				
				auto p_smaller = smaller.top();
				smaller.pop();

				p_greater.value -= (1.f - p_smaller.value);
				
				accept[p_smaller.id] = p_smaller.value;
				alias[p_smaller.id] = p_greater.id;

				if (p_greater.value == 1.f)
				{
					continue;
				}

				if (p_greater.value > 1.f)
				{
					greater.push(p_greater);
				}
				else
				{
					smaller.push(p_greater);
				}
			}

			while (!greater.empty())
			{
				auto p_greater = greater.top();
				greater.pop();

				accept[p_greater.id] = 1.f;
			}

			while (!smaller.empty())
			{
				auto p_smaller = smaller.top();
				smaller.pop();

				accept[p_smaller.id] = 1.f;
			}
		}

	public:
		float* accept;
		int* alias;
		int N;
	};
}
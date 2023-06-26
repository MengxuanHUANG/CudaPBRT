#include "alias.h"

#include "PBRT/pbrt.h"

namespace CudaPBRT
{
	Cuda_AlaisSampler_1D::Cuda_AlaisSampler_1D(int N, const float* distribution)
		:N(N)
	{
		CreateSampler(N, distribution);
	}

	Cuda_AlaisSampler_1D::~Cuda_AlaisSampler_1D()
	{
		FreeSampler();
	}

	void Cuda_AlaisSampler_1D::CreateSampler(int N, const float* distribution)
	{
		float* accept_data = new float[N];
		int* alias_data = new int[N];

		float* normalized_dis = new float[N];

		float sum = 0.f;
		for (int i = 0; i < N; ++i)
		{
			//if (distribution[i] > 10.f)
			//{
			//	printf("id: %d, value: %f", distribution[i]);
			//}
			sum += distribution[i];
		}
		
		sum /= static_cast<float>(N);

		for (int i = 0; i < N; ++i) normalized_dis[i] = distribution[i] / sum;


		AlaisSampler_1D::Create(N, normalized_dis, accept_data, alias_data);

		// mallocate memory and copy data to GPU
		BufferData<float>(device_distribution, distribution, N);
		BufferData<float>(device_accept, accept_data, N);
		BufferData<int>(device_alias, alias_data, N);

		delete[] normalized_dis;
		delete[] accept_data;
		delete[] alias_data;
	}

	void Cuda_AlaisSampler_1D::FreeSampler()
	{
		N = 0;
		CUDA_FREE(device_distribution);
		CUDA_FREE(device_accept);
		CUDA_FREE(device_alias);
	}
}
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
		//float* accept_data = new float[N];
		//int* alias_data = new int[N];

		std::vector<float> normalized_dis(N);
		std::vector<float> accept_data(N);
		std::vector<int> alias_data(N);

		float sum = 0.f;
		for (int i = 0; i < N; ++i)
		{
			sum += distribution[i];
		}
		
		sum /= static_cast<float>(N);

		for (int i = 0; i < N; ++i)
		{
			normalized_dis[i] = distribution[i] / sum;
		}

		AlaisSampler_1D::Create(N, normalized_dis.data(), accept_data.data(), alias_data.data());

		// mallocate memory and copy data to GPU
		BufferData<float>(device_distribution, normalized_dis.data(), N);
		BufferData<float>(device_accept, accept_data.data(), N);
		BufferData<int>(device_alias, alias_data.data(), N);
	}

	void Cuda_AlaisSampler_1D::FreeSampler()
	{
		N = 0;
		CUDA_FREE(device_distribution);
		CUDA_FREE(device_accept);
		CUDA_FREE(device_alias);
	}
}
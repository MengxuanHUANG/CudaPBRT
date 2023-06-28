#pragma once

#include "light.h"
#include "PBRT/Sampler/alias.h"

namespace CudaPBRT
{
	class EnvironmentLight : public Light
	{
#define m_EnvMap unionLightData.envLightData.envMap
#define m_Width unionLightData.envLightData.width
#define m_Height unionLightData.envLightData.height
#define m_Distribution unionLightData.envLightData.distribution
#define m_Accept unionLightData.envLightData.accept
#define m_Alias unionLightData.envLightData.alias
	public:
		CPU_GPU EnvironmentLight(const LightData& data)
		{
			m_EnvMap.m_TexObj = data.union_data.env_light.env_map;
			m_Width = data.union_data.env_light.width;
			m_Height = data.union_data.env_light.height;
			m_Accept = data.union_data.env_light.accept;
			m_Distribution = data.union_data.env_light.distribution;
			m_Alias = data.union_data.env_light.alias;
		}

		GPU_ONLY virtual Spectrum GetLe(const glm::vec3& p = glm::vec3(0.f)) const
		{
			float4 value = m_EnvMap.GetIrradiance(p);
			return 5.f * glm::clamp(Spectrum(value.x, value.y, value.z), 0.f, 50.f);
		}

		CPU_GPU virtual int GetShapeId() const 
		{ 
			return -1; 
		}

		GPU_ONLY virtual LightSample Sample_Li(const glm::vec3& p, const glm::vec3& normal, const glm::vec2& xi) const
		{
			int sampled_index = AlaisSampler_1D::Sample(xi, m_Width * m_Height, m_Accept, m_Alias);

			glm::ivec2 iuv = glm::ivec2(sampled_index % m_Width, 
										glm::floor(static_cast<float>(sampled_index) / static_cast<float>(m_Width)));

			glm::vec2 uv = glm::vec2(iuv) / glm::vec2(m_Width, m_Height);
			
			glm::vec3 sampled_p = m_EnvMap.GetWiWFromUV(uv);

			float theta = uv.x * Pi;
			float sinTheta = glm::sin(theta);
			float pdf = m_Distribution[sampled_index] / (2.f * Pi * Pi);

			if (sinTheta > 0.00001f)
			{
				pdf /= sinTheta;
			}

			return { this,
					 pdf,
					 sampled_p,
					 sampled_p};
		}

		GPU_ONLY virtual float PDF(const glm::vec3& p, const glm::vec3& wiW, float t, const glm::vec3& normal) const
		{
			return Sampler::SquareToSphereUniformPDF(wiW);
		}
#undef m_EnvMap
	};
}
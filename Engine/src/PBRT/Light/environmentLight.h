#pragma once

#include "light.h"

namespace CudaPBRT
{
	class EnvironmentLight : public Light
	{
#define m_EnvMap unionLightData.envLightData.envMap
	public:
		CPU_GPU EnvironmentLight(const LightData& data)
		{
			m_EnvMap.m_TexObj = data.union_data.environment_map;
		}

		GPU_ONLY virtual Spectrum GetLe(const glm::vec3& p = glm::vec3(0.f)) const
		{
			float4 value = m_EnvMap.GetIrradiance(p);
			return Spectrum(value.x, value.y, value.z);
		}

		CPU_GPU virtual int GetShapeId() const 
		{ 
			return -1; 
		}

		GPU_ONLY virtual LightSample Sample_Li(const glm::vec3& p, const glm::vec3& normal, const glm::vec2& xi) const
		{
			glm::vec3 sampled_p = Sampler::SquareToSphereUniform(xi);
			return { this,
					Sampler::SquareToSphereUniformPDF(sampled_p),
					sampled_p,
					glm::normalize(sampled_p)};
		}

		GPU_ONLY virtual float PDF(const glm::vec3& p, const glm::vec3& wiW, float t, const glm::vec3& normal) const
		{
			return Sampler::SquareToSphereUniformPDF(wiW);
		}
#undef m_EnvMap
	};
}
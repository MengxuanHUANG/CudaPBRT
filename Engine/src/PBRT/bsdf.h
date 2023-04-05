#pragma once
#include "PBRT/pbrtDefine.h"
#include "PBRT/Spectrum.h"
#include "PBRT/bxdfs.h"

#include <glm/glm.hpp>

namespace CudaPBRT
{
	class BSDF
	{
	public:
		CPU_GPU BSDF(BxDF* bxdf, float eta = 1.f)
			:eta(eta), m_BxDF(bxdf)
		{}

		CPU_GPU ~BSDF()
		{
			if (m_BxDF != nullptr)
			{
				delete m_BxDF;
			}
		}
		CPU_GPU Spectrum f(const Spectrum& R, const glm::vec3& wiW, const glm::vec3& woW, const glm::vec3& normal) const
		{
			glm::vec3 wo = WorldToLocal(normal) * woW;
			glm::vec3 wi = WorldToLocal(normal) * wiW;

			if (wo.z == 0.f)
			{
				return Spectrum(0.f);
			}

			return m_BxDF->f(R, wo, wi);
		}

		CPU_GPU BSDFSample Sample_f(const Spectrum& R, const glm::vec3& woW, const glm::vec3& normal, const glm::vec2& xi) const
		{
			glm::vec3 wo = WorldToLocal(normal) * woW;

			if (wo.z == 0.f)
			{
				return BSDFSample();
			}

			return m_BxDF->Sample_f(R, wo, normal, xi);
		}

		CPU_GPU float PDF(const glm::vec3& wiW, const glm::vec3& woW, const glm::vec3& normal) const
		{
			glm::vec3 wo = WorldToLocal(normal) * woW;
			glm::vec3 wi = WorldToLocal(normal) * wiW;

			return m_BxDF->PDF(wi, wo);
		}

	public:
		float eta;

	public:
		BxDF* m_BxDF;
	};
}

#pragma once
#include "PBRT/pbrtDefine.h"
#include "PBRT/spectrum.h"
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
		INLINE CPU_GPU Spectrum f(const Spectrum& R, const glm::vec3& woW, const glm::vec3& wiW, const glm::vec3& normal) const
		{
			glm::mat3 toLocal = WorldToLocal(normal);

			glm::vec3 wo = toLocal * woW;
			glm::vec3 wi = toLocal * wiW;

			return m_BxDF->f(R, wo, wi);
		}

		INLINE CPU_GPU BSDFSample Sample_f(const Spectrum& R, float etaA, const glm::vec3& woW, const glm::vec3& normal, const glm::vec2& xi) const
		{
			glm::vec3 wo = WorldToLocal(normal) * woW;

			return m_BxDF->Sample_f(R, etaA, wo, normal, xi);
		}

		INLINE CPU_GPU float PDF(const glm::vec3& woW, const glm::vec3& wiW, const glm::vec3& normal) const
		{
			glm::mat3 toLocal = WorldToLocal(normal);
			glm::vec3 wo = toLocal * woW;
			glm::vec3 wi = toLocal * wiW;

			return m_BxDF->PDF(wo, wi);
		}

	public:
		float eta;

	public:
		BxDF* m_BxDF;
	};
}

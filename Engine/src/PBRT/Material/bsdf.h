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
		CPU_GPU BSDF() {}
		CPU_GPU virtual ~BSDF() {}

		CPU_GPU virtual Spectrum f(const BSDFData& data, const glm::vec3& woW, const glm::vec3& wiW) const = 0;
		CPU_GPU virtual BSDFSample Sample_f(const BSDFData& data, const glm::vec3& woW, RNG& rng) const = 0;
		CPU_GPU virtual float PDF(const BSDFData& data, const glm::vec3& woW, const glm::vec3& wiW) const = 0;
	};

	class SingleBSDF : public BSDF
	{
	public:
		CPU_GPU SingleBSDF(BxDF* bxdf)
			:m_BxDF(bxdf)
		{}

		CPU_GPU ~SingleBSDF()
		{
			SAFE_FREE(m_BxDF);
		}

		CPU_GPU virtual Spectrum f(const BSDFData& data, const glm::vec3& woW, const glm::vec3& wiW) const override
		{
			glm::mat3 toLocal = WorldToLocal(data.normal);

			glm::vec3 wo = glm::normalize(toLocal * woW);
			glm::vec3 wi = glm::normalize(toLocal * wiW);

			return m_BxDF->f(data, wo, wi);
		}

		CPU_GPU virtual BSDFSample Sample_f(const BSDFData& data, const glm::vec3& woW, RNG& rng) const override
		{
			glm::vec3 wo = glm::normalize(WorldToLocal(data.normal) * woW);

			return m_BxDF->Sample_f(data, wo, rng);
		}

		CPU_GPU virtual float PDF(const BSDFData& data, const glm::vec3& woW, const glm::vec3& wiW) const override
		{
			glm::mat3 toLocal = WorldToLocal(data.normal);
			glm::vec3 wo = glm::normalize(toLocal * woW);
			glm::vec3 wi = glm::normalize(toLocal * wiW);

			return m_BxDF->PDF(data, wo, wi);
		}

	public:
		BxDF* m_BxDF;
	};

	class FresnelBSDF : public BSDF
	{
	public:
		CPU_GPU FresnelBSDF(BxDF* bxdf_1, BxDF* bxdf_2, float eta)
		{
			etaB = eta;
			m_BxDFs[0] = bxdf_1;
			m_BxDFs[1] = bxdf_2;
		}
		
		CPU_GPU ~FresnelBSDF()
		{
			SAFE_FREE(m_BxDFs[0]);
			SAFE_FREE(m_BxDFs[1]);
		}

		CPU_GPU virtual Spectrum f(const BSDFData& data, const glm::vec3& woW, const glm::vec3& wiW) const
		{
			glm::mat3 toLocal = WorldToLocal(data.normal);

			glm::vec3 wo = glm::normalize(toLocal * woW);
			glm::vec3 wi = glm::normalize(toLocal * wiW);

			return m_BxDFs[0]->f(data, wo, wi);
		}

		CPU_GPU virtual BSDFSample Sample_f(const BSDFData& data, const glm::vec3& woW, RNG& rng) const
		{
			glm::vec3 wo = glm::normalize(WorldToLocal(data.normal) * woW);
			const float etaA = data.eta;

			float F = FresnelDielectric(etaA, etaB, CosTheta(wo));
			if (rng.rand() < F)
			{
				return m_BxDFs[0]->Sample_f(data, wo, rng);
			}
			else
			{
				return m_BxDFs[1]->Sample_f(data, wo, rng);
			}
		}

		CPU_GPU virtual float PDF(const BSDFData& data, const glm::vec3& woW, const glm::vec3& wiW) const
		{
			glm::mat3 toLocal = WorldToLocal(data.normal);
			glm::vec3 wo = glm::normalize(toLocal * woW);
			glm::vec3 wi = glm::normalize(toLocal * wiW);

			return m_BxDFs[0]->PDF(data, wo, wi);
		}

	protected:
		float etaB;
		BxDF* m_BxDFs[2];
	};
}
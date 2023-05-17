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

		CPU_GPU virtual Spectrum f(const Spectrum& R, const glm::vec3& woW, const glm::vec3& wiW, const glm::vec3& normal) const = 0;
		CPU_GPU virtual BSDFSample Sample_f(const Spectrum& R, float etaA, const glm::vec3& woW, const glm::vec3& normal, const glm::vec2& xi) const = 0;
		CPU_GPU virtual float PDF(const glm::vec3& woW, const glm::vec3& wiW, const glm::vec3& normal) const = 0;
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

		CPU_GPU virtual Spectrum f(const Spectrum& R, const glm::vec3& woW, const glm::vec3& wiW, const glm::vec3& normal) const override
		{
			glm::mat3 toLocal = WorldToLocal(normal);

			glm::vec3 wo = toLocal * woW;
			glm::vec3 wi = toLocal * wiW;

			return m_BxDF->f(R, wo, wi);
		}

		CPU_GPU virtual BSDFSample Sample_f(const Spectrum& R, float etaA, const glm::vec3& woW, const glm::vec3& normal, const glm::vec2& xi) const override
		{
			glm::vec3 wo = WorldToLocal(normal) * woW;

			return m_BxDF->Sample_f(R, etaA, wo, normal, xi);
		}

		CPU_GPU virtual float PDF(const glm::vec3& woW, const glm::vec3& wiW, const glm::vec3& normal) const override
		{
			glm::mat3 toLocal = WorldToLocal(normal);
			glm::vec3 wo = toLocal * woW;
			glm::vec3 wi = toLocal * wiW;

			return m_BxDF->PDF(wo, wi);
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

		CPU_GPU virtual Spectrum f(const Spectrum& R, const glm::vec3& woW, const glm::vec3& wiW, const glm::vec3& normal) const
		{
			glm::mat3 toLocal = WorldToLocal(normal);

			glm::vec3 wo = toLocal * woW;
			glm::vec3 wi = toLocal * wiW;

			return m_BxDFs[0]->f(R, wo, wi);
		}

		CPU_GPU virtual BSDFSample Sample_f(const Spectrum& R, float etaA, const glm::vec3& woW, const glm::vec3& normal, const glm::vec2& xi) const
		{
			glm::vec3 wo = WorldToLocal(normal) * woW;
			
			float F = FresnelDielectric(etaA, etaB, CosTheta(wo));
			if (xi.x < F)
			{
				return m_BxDFs[0]->Sample_f(R, etaA, wo, normal, xi);
			}
			else
			{
				return m_BxDFs[1]->Sample_f(R, etaA, wo, normal, xi);
			}
		}

		CPU_GPU virtual float PDF(const glm::vec3& woW, const glm::vec3& wiW, const glm::vec3& normal) const
		{
			glm::mat3 toLocal = WorldToLocal(normal);
			glm::vec3 wo = toLocal * woW;
			glm::vec3 wi = toLocal * wiW;

			return m_BxDFs[0]->PDF(wo, wi);
		}

	protected:
		float etaB;
		BxDF* m_BxDFs[2];
	};
}

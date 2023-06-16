#pragma once
#include "PBRT/pbrtDefine.h"
#include "PBRT/spectrum.h"
#include "PBRT/bxdfs.h"

#include <glm/glm.hpp>

namespace CudaPBRT
{
	struct GeneralBSDFData
	{
		float etaB;
		BxDF bxdfs[2];
	};

	union BSDFUnionData
	{
		GeneralBSDFData general_data;
	};

	class BSDF
	{
	public:
		CPU_GPU BSDF() {}
		CPU_GPU virtual Spectrum f(const BSDFData& data, const glm::vec3& wo, const glm::vec3& wi) const { return Spectrum(0.f); }
		CPU_GPU virtual BSDFSample Sample_f(const BSDFData& data, const glm::vec3& wo, RNG& rng) const { return BSDFSample(); }
		CPU_GPU virtual float PDF(const BSDFData& data, const glm::vec3& wo, const glm::vec3& wi) const { return 0.f; }
	
	public:
		GeneralBSDFData m_GeneralData;
	};

	class SingleBSDF : public BSDF
	{
#define m_BxDF m_GeneralData.bxdfs[0]
	public:
		CPU_GPU virtual Spectrum f(const BSDFData& data, const glm::vec3& wo, const glm::vec3& wi) const override
		{
			return m_BxDF.f(data, wo, wi);
		}

		CPU_GPU virtual BSDFSample Sample_f(const BSDFData& data, const glm::vec3& wo, RNG& rng) const override
		{
			return m_BxDF.Sample_f(data, wo, rng);
		}

		CPU_GPU virtual float PDF(const BSDFData& data, const glm::vec3& wo, const glm::vec3& wi) const override
		{
			return m_BxDF.PDF(data, wo, wi);
		}
#undef m_BxDF
	};

	class FresnelBSDF : public BSDF
	{
	public:
#define m_BxDFs m_GeneralData.bxdfs
#define etaB m_GeneralData.etaB
		CPU_GPU virtual Spectrum f(const BSDFData& data, const glm::vec3& wo, const glm::vec3& wi) const
		{
			return m_BxDFs[0].f(data, wo, wi);
		}

		CPU_GPU virtual BSDFSample Sample_f(const BSDFData& data, const glm::vec3& wo, RNG& rng) const
		{
			const float etaA = data.eta;

			float F = FresnelDielectric(etaA, etaB, CosTheta(wo));
			if (rng.rand() < F)
			{
				return m_BxDFs[0].Sample_f(data, wo, rng);
			}
			else
			{
				return m_BxDFs[1].Sample_f(data, wo, rng);
			}
		}

		CPU_GPU virtual float PDF(const BSDFData& data, const glm::vec3& wo, const glm::vec3& wi) const
		{
			return m_BxDFs[0].PDF(data, wo, wi);
		}
#undef m_BxDFs
#undef etaB
	};
}

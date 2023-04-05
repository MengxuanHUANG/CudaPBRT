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
			//if (m_BxDF != nullptr)
			//{
			//	delete m_BxDF;
			//}
		}
		CPU_GPU Spectrum f(const glm::vec3& wiW, const glm::vec3& woW) const
		{
			glm::vec3 wi, wo;
			// TODO: transform wiW, woW from world to local

			if (wo.z == 0.f)
			{
				return Spectrum(0.f);
			}

			// TODO: compute f from bxdf
		}

		CPU_GPU BSDFSample Sample_f(const glm::vec3& woW, const glm::vec3& normal, const glm::vec2& xi) const
		{
			glm::vec3 wi, wo;
			wo = WorldToLocal(normal) * woW;

			if (wo.z == 0.f)
			{
				return BSDFSample();
			}

			return m_BxDF->Sample_f(wo, normal, xi);
		}

		CPU_GPU float PDF(const glm::vec3& wiW, const glm::vec3& woW) const
		{
			glm::vec3 wi, wo;
			// TODO: transform wiW, woW from world to local

			return m_BxDF->PDF(wi, wo);
		}

	public:
		float eta;

	public:
		BxDF* m_BxDF;
	};
}

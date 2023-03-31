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
		GPU_ONLY BSDF(BxDF bxdf)
			:m_Bxdf(bxdf)
		{}

		GPU_ONLY Spectrum f(const glm::vec3& wiW, const glm::vec3& woW) const
		{
			glm::vec3 wi, wo;
			// TODO: transform wiW, woW from world to local

			if (wo.z == 0.f)
			{
				return Spectrum(0.f);
			}

			// TODO: compute f from bxdf
		}

		GPU_ONLY BSDFSample Sample_f(const glm::vec3& woW) const
		{
			glm::vec3 wi, wo;
			// TODO: transform wiW, woW from world to local

			if (wo.z == 0.f)
			{
				return Spectrum(0.f);
			}

			// TODO: compute f from bxdf
			BSDFSample bsdfSample = m_Bxdf.Sample_f(wo);

			return bsdfSample;
		}

		GPU_ONLY float PDF(const glm::vec3& wiW, const glm::vec3& woW) const
		{
			glm::vec3 wi, wo;
			// TODO: transform wiW, woW from world to local

			return m_Bxdf.PDF(wi, wo);
		}

	private:
		BxDF m_Bxdf;
	};
}

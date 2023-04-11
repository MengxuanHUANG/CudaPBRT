#include "bxdfs.h"

namespace CudaPBRT
{
	CPU_GPU float FresnelDielectric(float etaI, float etaT, float cosThetaI)
	{
		cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

		float sinThetaI = glm::sqrt(glm::max(0.f, 1.f - cosThetaI * cosThetaI));
		float sinThetaT = etaI / etaT * sinThetaI;

		if (sinThetaT >= 1.f)
		{
			return 1.f;
		}

		float cosThetaT = glm::sqrt(glm::max(0.f, 1.f - sinThetaT * sinThetaT));
		float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
		float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));

		return (Rparl * Rparl + Rperp * Rperp) / 2.f;
	}
}
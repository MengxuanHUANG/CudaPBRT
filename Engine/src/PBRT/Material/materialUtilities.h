#pragma once

#include "PBRT/pbrtDefine.h"
#include "PBRT/pbrtUtilities.h"

namespace CudaPBRT
{
	// bsdf help functions
	INLINE CPU_GPU float FresnelDielectric(float etaI, float etaT, float cosThetaI)
	{
		cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

		if (cosThetaI < 0.f)
		{
			float temp = etaI;
			etaI = etaT;
			etaT = temp;
			cosThetaI = -cosThetaI;
		}

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

	INLINE CPU_GPU glm::vec3 Reflect(const glm::vec3& wo)
	{
		return { -wo.x, -wo.y, wo.z };
	}

	INLINE CPU_GPU bool Refract(const glm::vec3& wi, const glm::vec3& n, float eta, glm::vec3& wt)
	{
		float cosThetaI = glm::dot(n, wi);
		float sin2ThetaI = glm::max(0.f, 1.f - cosThetaI * cosThetaI);
		float sin2ThetaT = eta * eta * sin2ThetaI;

		if (sin2ThetaT >= 1.f) return false;

		float cosThetaT = std::sqrt(1 - sin2ThetaT);
		wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
		return true;
	}

	INLINE CPU_GPU glm::vec3 Faceforward(const glm::vec3& n, const glm::vec3& v)
	{
		return (glm::dot(n, v) < 0.f) ? -n : n;
	}

	INLINE CPU_GPU bool SameHemisphere(const glm::vec3& w, const glm::vec3& wp) {
		return w.z * wp.z > 0;
	}

	INLINE CPU_GPU glm::vec3 Sample_wh(const glm::vec3& wo, const glm::vec2& xi, float roughness)
	{
		glm::vec3 wh;

		float cosTheta = 0.f;
		float phi = 2.f * Pi * xi[1];

		float tanTheta2 = roughness * roughness * xi[0] / (1.0f - xi[0]);
		cosTheta = 1.f / glm::sqrt(1.f + tanTheta2);

		float sinTheta = glm::sqrt(glm::max(0.f, 1.f - cosTheta * cosTheta));

		wh = glm::vec3(sinTheta * glm::cos(phi), sinTheta * glm::sin(phi), cosTheta);
		if (!SameHemisphere(wo, wh)) wh = -wh;

		return wh;
	}

	INLINE CPU_GPU float TrowbridgeReitzD(const glm::vec3& wh, float roughness) 
	{
		float tan2Theta = Tan2Theta(wh);
		if (isinf(tan2Theta)) return 0.f;

		float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);

		float e = (Cos2Phi(wh) / (roughness * roughness) + Sin2Phi(wh) / (roughness * roughness)) * tan2Theta;
		return 1 / (Pi * roughness * roughness * cos4Theta * (1 + e) * (1 + e));
	}

	INLINE CPU_GPU float Lambda(const glm::vec3& w, float roughness) 
	{
		float absTanTheta = abs(TanTheta(w));
		if (isinf(absTanTheta)) return 0.;

		// Compute alpha for direction w
		float alpha = glm::sqrt(Cos2Phi(w) * roughness * roughness + Sin2Phi(w) * roughness * roughness);
		float alpha2Tan2Theta = (roughness * absTanTheta) * (roughness * absTanTheta);
		return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
	}

	INLINE CPU_GPU float TrowbridgeReitzG(const glm::vec3& wo, const glm::vec3& wi, float roughness) 
	{
		return 1.f / (1.f + Lambda(wo, roughness) + Lambda(wi, roughness));
	}

	INLINE CPU_GPU float TrowbridgeReitzPdf(const glm::vec3& wo, const glm::vec3& wh, float roughness) 
	{
		return TrowbridgeReitzD(wh, roughness) * AbsCosTheta(wh);
	}
}
#pragma once

#include "PBRT/pbrtDefine.h"
#include "PBRT/pbrtUtilities.h"

#include "PBRT/Sampler/sampler.h"
#include "PBRT/spectrum.h"

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
		return w.z * wp.z > 0.f;
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
		if (glm::isinf(tan2Theta)) return 0.f;

		float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
		float alpha = roughness * roughness;
		float e = (Cos2Phi(wh) / alpha + Sin2Phi(wh) / alpha) * tan2Theta;
		return 1.f / (Pi * alpha * cos4Theta * (1.f + e) * (1.f + e));
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

	INLINE CPU_GPU float SchlickWeight(float cosTheta) 
	{
		float m = glm::clamp(1.f - cosTheta, 0.f, 1.f);
		return (m * m) * (m * m) * m; // m ^ 5
	}
	
	INLINE CPU_GPU float FrSchlick(float R0, const float& cosTheta)
	{
		return glm::mix(SchlickWeight(cosTheta), R0, 1.f);
	}

	INLINE CPU_GPU Spectrum FrSchlick(const Spectrum& R0, const float& cosTheta)
	{
		return glm::mix(R0, Spectrum(1.f), SchlickWeight(cosTheta));
	}

	INLINE CPU_GPU float SchlickR0FromEta(const float& eta) { return glm::sqrt(eta - 1.f) / glm::sqrt(eta + 1.f); }

	INLINE CPU_GPU float SchlickG(const float& cosTheta, const float& alpha)
	{
		float a = alpha * 0.5f;
		return cosTheta / (cosTheta * (1.f - a) + a);
	}

	INLINE CPU_GPU float SmithG(const float& cosWo, const float& cosWi, const float& alpha)
	{
		return SchlickG(glm::abs(cosWo), alpha) * SchlickG(glm::abs(cosWi), alpha);
	}

	INLINE CPU_GPU float GTR2Distrib(const float& cosTheta, const float& alpha)
	{
		float NdotH = glm::max(cosTheta, 0.f);
		float denom = NdotH * NdotH * (alpha - 1.f) + 1.f;
		denom = denom * denom * Pi;
		return alpha / denom;
	}

	INLINE CPU_GPU float GTR2Pdf(glm::vec3 normal, glm::vec3 whW, glm::vec3 woW, const float& alpha)
	{
		return GTR2Distrib(glm::dot(normal, whW), alpha) * SchlickG(glm::dot(normal, woW), alpha) *
			AbsDot(whW, woW) / AbsDot(normal, woW);
	}

	INLINE CPU_GPU glm::vec3 GTR2Sample(glm::vec3 normal, glm::vec3 wo, const float& alpha, glm::vec2 xi)
	{
		glm::vec3 vh = glm::normalize(wo * glm::vec3(alpha, alpha, 1.f));
		float lenSq = vh.x * vh.x + vh.y * vh.y;
		glm::vec3 t = lenSq > 0.f ? glm::vec3(-vh.y, vh.x, 0.f) / glm::sqrt(lenSq) : glm::vec3(1.f, 0.f, 0.f);
		glm::vec3 b = glm::cross(vh, t);

		glm::vec2 p = Sampler::SquareToDiskConcentric(xi);
		float s = 0.5f * (vh.z + 1.f);
		p.y = (1.f - s) * glm::sqrt(1.f - p.x * p.x) + s * p.y;

		glm::vec3 h = t * p.x + b * p.y + vh * glm::sqrt(glm::max(0.f, 1.f - glm::dot(p, p)));
		h = glm::vec3(h.x * alpha, h.y * alpha, glm::max(0.f, h.z));
		return h;
	}
}
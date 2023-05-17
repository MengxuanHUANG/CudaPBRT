#pragma once

#include "Core/Core.h"
#include "pbrtDefine.h"
#include <glm/glm.hpp>

namespace CudaPBRT
{
	INLINE CPU_GPU float AbsDot(const glm::vec3& wi, const glm::vec3& nor) { return glm::abs(glm::dot(wi, nor)); }

	INLINE CPU_GPU float CosTheta(const glm::vec3& w) { return w.z; }
	INLINE CPU_GPU float AbsCosTheta(const glm::vec3& w) { return glm::abs(w.z); }
	INLINE CPU_GPU  float PowerHeuristic(int nf, float fPdf, int ng, float gPdf)
	{
		float f = static_cast<float>(nf) * fPdf;
		float g = static_cast<float>(ng) * gPdf;

		return (f * f) / (f * f + g * g);
	}

	INLINE CPU_GPU void coordinateSystem(const glm::vec3& v1, glm::vec3& v2, glm::vec3& v3)
	{
		if (glm::abs(v1.x) > glm::abs(v1.y))
			v2 = glm::vec3(-v1.z, 0, v1.x) / glm::sqrt(v1.x * v1.x + v1.z * v1.z);
		else
			v2 = glm::vec3(0, v1.z, -v1.y) / glm::sqrt(v1.y * v1.y + v1.z * v1.z);
		v3 = glm::cross(v1, v2);
	}

	INLINE CPU_GPU glm::mat3 LocalToWorld(const::glm::vec3& nor)
	{
		glm::vec3 tan, bit;
		coordinateSystem(nor, tan, bit);
		return glm::mat3(tan, bit, nor);
	}
	INLINE CPU_GPU glm::mat3 WorldToLocal(const::glm::vec3& nor)
	{
		return glm::transpose(LocalToWorld(nor));
	}

	template<typename T>
	INLINE CPU_GPU T BarycentricInterpolation(const T& c1, const T& c2, const T& c3, float u, float v)
	{
		return u * c1 + v * c2 + (1.f - u - v) * c3;
	}

	GPU_ONLY float4 ReadTexture(const CudaTexObj& tex_obj, const glm::vec2& uv);

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
}
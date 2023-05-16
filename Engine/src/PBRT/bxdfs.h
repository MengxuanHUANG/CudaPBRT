#pragma once
#include "PBRT/pbrtDefine.h"
#include "PBRT/Spectrum.h"

#include <glm/glm.hpp>

#include "Sampler/sampler.h"

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

	struct BSDFSample
	{
		Spectrum f;
		glm::vec3 wiW;
		float pdf = 0.f;
		float eta = AirETA;

		CPU_GPU BSDFSample()
		{}

		CPU_GPU BSDFSample(const Spectrum& f, const glm::vec3& wiW, float pdf, float eta)
			:f(f), wiW(wiW), pdf(pdf), eta(eta)
		{
		}
	};

	class BxDF
	{
	public:
		CPU_GPU virtual Spectrum f(const Spectrum& R, const glm::vec3& wo, const glm::vec3& wi) const = 0;
		CPU_GPU virtual BSDFSample Sample_f(const Spectrum& R, float etaA, const glm::vec3& wo, const glm::vec3& normal, const glm::vec2& xi) const = 0;
		CPU_GPU virtual float PDF(const glm::vec3& wo, const glm::vec3& wi) const = 0;
	};

	class LambertianReflection : public BxDF
	{
	public:
		CPU_GPU virtual Spectrum f(const Spectrum& R, const glm::vec3& wo, const glm::vec3& wi) const override
		{
			return R * InvPi;
		}

		CPU_GPU virtual BSDFSample Sample_f(const Spectrum& R, float etaA, const glm::vec3& wo, const glm::vec3& normal, const glm::vec2& xi) const override
		{
			if (wo.z <= 0.f)
			{
				return BSDFSample();
			}
			glm::vec3 wi = Sampler::SquareToHemisphereCosine(xi);
			glm::vec3 wiW = glm::normalize(LocalToWorld(normal) * wi);
			
			return BSDFSample(f(R, wo, wi), wiW, PDF(wo, wi), etaA);
		}

		CPU_GPU virtual float PDF(const glm::vec3& wo, const glm::vec3& wi) const override
		{
			return Sampler::SquareToHemisphereCosinePDF(wi);
		}
	};

	class SpecularReflection : public BxDF
	{
		CPU_GPU virtual Spectrum f(const Spectrum& R, const glm::vec3& wo, const glm::vec3& wi) const override
		{
			return Spectrum(0.f);
		}

		CPU_GPU virtual BSDFSample Sample_f(const Spectrum& R, float etaA, const glm::vec3& wo, const glm::vec3& normal, const glm::vec2& xi) const override
		{
			glm::vec3 wi = Reflect(wo);

			glm::vec3 wiW = glm::normalize(LocalToWorld(normal) * wi);

			return BSDFSample(R / AbsCosTheta(wi), wiW, 1.f, etaA);
		}

		CPU_GPU virtual float PDF(const glm::vec3& wo, const glm::vec3& wi) const override
		{
			return 0.f;
		}
	};

	class SpecularTransmission : public BxDF
	{
	public:
		CPU_GPU SpecularTransmission(float eta)
			:etaB(eta)
		{}
		CPU_GPU virtual Spectrum f(const Spectrum& R, const glm::vec3& wo, const glm::vec3& wi) const override
		{
			return Spectrum(0.f);
		}

		CPU_GPU virtual BSDFSample Sample_f(const Spectrum& T, float etaA, const glm::vec3& wo, const glm::vec3& normal, const glm::vec2& xi) const override
		{
			bool entering = CosTheta(wo) > 0.f;
			float etaI = entering ? etaA : etaB;
			float etaT = entering ? etaB : etaA;
			
			glm::vec3 wi;

			if (!Refract(wo, Faceforward(glm::vec3(0, 0, 1), wo), etaI / etaT, wi))
			{
				return BSDFSample();
			}

			Spectrum ft = T;

			glm::vec3 wiW = glm::normalize(LocalToWorld(normal) * wi);

			return BSDFSample(ft / AbsCosTheta(wi), wiW, 1.f, etaB);
		}

		CPU_GPU virtual float PDF(const glm::vec3& wo, const glm::vec3& wi) const override
		{
			return 0.f;
		}

	public:
		float etaB;
	};

	class GlassBxDF : public BxDF
	{
	public:
		CPU_GPU GlassBxDF(float eta)
			:etaB(eta)
		{}
		CPU_GPU virtual Spectrum f(const Spectrum& R, const glm::vec3& wo, const glm::vec3& wi) const override
		{
			return Spectrum(0.f);
		}

		CPU_GPU virtual BSDFSample Sample_f(const Spectrum& T, float etaA, const glm::vec3& wo, const glm::vec3& normal, const glm::vec2& xi) const override
		{
			bool entering = CosTheta(wo) > 0.f;
			float etaI = entering ? etaA : etaB;
			float etaT = entering ? etaB : etaA;

			float F = FresnelDielectric(etaA, etaB, CosTheta(wo));
			if (xi.x < F)
			{
				// reflection
				glm::vec3 wi = Reflect(wo);
				glm::vec3 wiW = normalize(LocalToWorld(normal) * wi);

				return BSDFSample(T / AbsCosTheta(wi), wiW, 1.f, etaA);
			}
			else
			{
				// transimission
				glm::vec3 wi;

				if (!Refract(wo, Faceforward(glm::vec3(0, 0, 1), wo), etaI / etaT, wi))
				{
					// pure reflection
					return BSDFSample();
				}

				Spectrum ft = T;
				ft *= (etaI * etaI) / (etaT * etaT);
				glm::vec3 wiW = glm::normalize(LocalToWorld(normal) * wi);

				return BSDFSample(ft / AbsCosTheta(wi), wiW, 1.f, etaB);
			}
		}

		CPU_GPU virtual float PDF(const glm::vec3& wo, const glm::vec3& wi) const override
		{
			return 0.f;
		}

	public:
		float etaB;
	};
}
#pragma once
#include "pbrtDefine.h"
#include "pbrtUtilities.h"
#include "spectrum.h"

#include <glm/glm.hpp>

#include "Sampler/sampler.h"

namespace CudaPBRT
{
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

	// OrenNayar microfacet reflection
	class MicrofacetReflection : public BxDF
	{
	public:
		CPU_GPU virtual Spectrum f(const Spectrum& R, const glm::vec3& wo, const glm::vec3& wi) const override
		{
			float roughness = 0.001f;

			float cosThetaO = AbsCosTheta(wo);
			float cosThetaI = AbsCosTheta(wi);
			glm::vec3 wh = wi + wo;
			// Handle degenerate cases for microfacet reflection
			if (cosThetaI == 0 || cosThetaO == 0) return Spectrum();
			if (wh.x == 0 && wh.y == 0 && wh.z == 0) return Spectrum();
			wh = glm::normalize(wh);
			float F = 1;
			float D = TrowbridgeReitzD(wh, roughness);
			float G = TrowbridgeReitzG(wo, wi, roughness);

			return R * F * D * G / (4.f * cosThetaI * cosThetaO);
		}

		CPU_GPU virtual BSDFSample Sample_f(const Spectrum& R, float etaA, const glm::vec3& wo, const glm::vec3& normal, const glm::vec2& xi) const override
		{
			float roughness = 0.001f;
			if (wo.z <= 0.f)
			{
				return BSDFSample();
			}
			glm::vec3 wh = Sample_wh(wo, xi, roughness);
			glm::vec3 wi = glm::reflect(-wo, wh);

			glm::vec3 wiW = glm::normalize(LocalToWorld(normal) * wi);
			
			if (!SameHemisphere(wo, wi)) return BSDFSample();

			// Compute PDF of _wi_ for microfacet reflection
			float pdf = TrowbridgeReitzPdf(wo, wh, roughness) / (4 * glm::dot(wo, wh));

			return BSDFSample(f(R, wo, wi), wiW, pdf, etaA);
		}

		CPU_GPU virtual float PDF(const glm::vec3& wo, const glm::vec3& wi) const override
		{
			float roughness = 0.001f;
			glm::vec3 wh = glm::normalize(wo + wi);
			return TrowbridgeReitzPdf(wo, wh, roughness) / (4 * glm::dot(wo, wh));
		}
	};
}
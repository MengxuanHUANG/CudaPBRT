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

	// microfacet reflection
	class OrenNayar : public BxDF
	{
	public:
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
}
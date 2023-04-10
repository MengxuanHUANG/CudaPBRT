#pragma once
#include "PBRT/pbrtDefine.h"
#include "PBRT/Spectrum.h"

#include <glm/glm.hpp>

#include "Sampler/sampler.h"

namespace CudaPBRT
{
	struct BSDFSample
	{
		Spectrum f;
		glm::vec3 wiW;
		float pdf = 0.f;
		float eta = 1.f;

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
			glm::vec3 wi = Sampler::SquareToHemisphereCosine(xi);
			glm::vec3 wiW = normalize(LocalToWorld(normal) * wi);
			
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
			glm::vec3 wi = -wo;
			wi.z *= -1.f;

			glm::vec3 wiW = normalize(LocalToWorld(normal) * wi);

			return BSDFSample(R, wiW, 1.f, etaA);
		}

		CPU_GPU virtual float PDF(const glm::vec3& wo, const glm::vec3& wi) const override
		{
			return 0.f;
		}
	};
}
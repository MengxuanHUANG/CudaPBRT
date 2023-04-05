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
		CPU_GPU virtual Spectrum f(const Spectrum& R, const glm::vec3& wi, const glm::vec3& wo) const = 0;
		CPU_GPU virtual BSDFSample Sample_f(const Spectrum& R, const glm::vec3& wo, const glm::vec3& normal, const glm::vec2& xi) const = 0;
		CPU_GPU virtual float PDF(const glm::vec3& wi, const glm::vec3& wo) const = 0;
	};

	class LambertianReflection : public BxDF
	{
	public:
		CPU_GPU LambertianReflection(float eta)
			:eta(eta)
		{}
		CPU_GPU virtual Spectrum f(const Spectrum& R, const glm::vec3& wi, const glm::vec3& wo) const override
		{
			return R *InvPi;
		}

		CPU_GPU virtual BSDFSample Sample_f(const Spectrum& R, const glm::vec3& wo, const glm::vec3& normal, const glm::vec2& xi) const override
		{
			glm::vec3 wi = Sampler::SquareToHemisphereCosine(xi);
			glm::vec3 wiW = normalize(LocalToWorld(normal) * wi);
			
			return BSDFSample(f(R, wi, wo), wiW, PDF(wi, wo), eta);
		}

		CPU_GPU virtual float PDF(const glm::vec3& wi, const glm::vec3& wo) const override
		{
			return Sampler::SquareToHemisphereCosinePDF(wi);
		}

	public:
		float eta;
		Spectrum R;
	};
}
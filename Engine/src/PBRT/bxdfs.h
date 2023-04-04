#pragma once
#include "PBRT/pbrtDefine.h"
#include "PBRT/Spectrum.h"

#include <glm/glm.hpp>

namespace CudaPBRT
{
	struct BSDFSample
	{
	public:
		Spectrum f;
		glm::vec3 wi;
		float pdf = 0.f;
		float eta = 1.f;

		int type;
	};

	class BxDF
	{
	public:
		CPU_GPU virtual Spectrum f(const glm::vec3& wi, const glm::vec3& wo) const = 0;
		CPU_GPU virtual BSDFSample Sample_f(const glm::vec3& wo) const = 0;
		CPU_GPU virtual float PDF(const glm::vec3& wi, const glm::vec3& wo) const = 0;
	};

	class LambertianReflection : public BxDF
	{
	public:
		CPU_GPU virtual Spectrum f(const glm::vec3& wi, const glm::vec3& wo) const override
		{
			return Spectrum(0) / InvPi;
		}

		CPU_GPU virtual BSDFSample Sample_f(const glm::vec3& wo) const override
		{
			// TODO: sample the hemisphere

			return BSDFSample();
		}

		CPU_GPU virtual float PDF(const glm::vec3& wi, const glm::vec3& wo) const override
		{
			return 0.f;
		}
	};
}
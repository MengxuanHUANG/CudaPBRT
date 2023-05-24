#pragma once
#include "pbrtDefine.h"
#include "pbrtUtilities.h"
#include "spectrum.h"

#include <glm/glm.hpp>

#include "Sampler/sampler.h"
#include "Sampler/rng.h"

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

	struct BSDFData
	{
		const glm::vec3& normal;
		const float& roughness;
		const float& metallic;
		const float& eta;
		const Spectrum& R;

		CPU_GPU BSDFData(const glm::vec3& n, const float& r, const float& m, const float& eta, const Spectrum& R)
			:normal(n), roughness(r), metallic(m), eta(eta), R(R)
		{}
	};

	class BxDF
	{
	public:
		CPU_GPU virtual Spectrum f(const BSDFData& data, const glm::vec3& wo, const glm::vec3& wi) const = 0;
		CPU_GPU virtual BSDFSample Sample_f(const BSDFData& data, const glm::vec3& wo, RNG& rng) const = 0;
		CPU_GPU virtual float PDF(const BSDFData& data, const glm::vec3& wo, const glm::vec3& wi) const = 0;
	};

	class LambertianReflection : public BxDF
	{
	public:
		CPU_GPU virtual Spectrum f(const BSDFData& data, const glm::vec3& wo, const glm::vec3& wi) const override
		{
			return data.R * InvPi;
		}

		CPU_GPU virtual BSDFSample Sample_f(const BSDFData& data, const glm::vec3& wo, RNG& rng) const override
		{
			if (wo.z <= 0.f)
			{
				return BSDFSample();
			}
			glm::vec3 wi = Sampler::SquareToHemisphereCosine({rng.rand(), rng.rand()});
			glm::vec3 wiW = glm::normalize(LocalToWorld(data.normal) * wi);
			
			return BSDFSample(f(data, wo, wi), wiW, PDF(data, wo, wi), data.eta);
		}

		CPU_GPU virtual float PDF(const BSDFData& data, const glm::vec3& wo, const glm::vec3& wi) const override
		{
			return Sampler::SquareToHemisphereCosinePDF(wi);
		}
	};

	class SpecularReflection : public BxDF
	{
		CPU_GPU virtual Spectrum f(const BSDFData& data, const glm::vec3& wo, const glm::vec3& wi) const override
		{
			return Spectrum(0.f);
		}

		CPU_GPU virtual BSDFSample Sample_f(const BSDFData& data, const glm::vec3& wo, RNG& rng) const override
		{
			glm::vec3 wi = Reflect(wo);

			glm::vec3 wiW = glm::normalize(LocalToWorld(data.normal) * wi);

			return BSDFSample(data.R / AbsCosTheta(wi), wiW, 1.f, data.eta);
		}

		CPU_GPU virtual float PDF(const BSDFData& data, const glm::vec3& wo, const glm::vec3& wi) const override
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
		CPU_GPU virtual Spectrum f(const BSDFData& data, const glm::vec3& wo, const glm::vec3& wi) const override
		{
			return Spectrum(0.f);
		}

		CPU_GPU virtual BSDFSample Sample_f(const BSDFData& data, const glm::vec3& wo, RNG& rng) const override
		{
			const float etaA = data.eta;

			bool entering = CosTheta(wo) > 0.f;
			float etaI = entering ? etaA : etaB;
			float etaT = entering ? etaB : etaA;
			
			glm::vec3 wi;

			if (!Refract(wo, Faceforward(glm::vec3(0, 0, 1), wo), etaI / etaT, wi))
			{
				return BSDFSample();
			}

			Spectrum ft = data.R;

			glm::vec3 wiW = glm::normalize(LocalToWorld(data.normal) * wi);

			return BSDFSample(ft / AbsCosTheta(wi), wiW, 1.f, etaB);
		}

		CPU_GPU virtual float PDF(const BSDFData& data, const glm::vec3& wo, const glm::vec3& wi) const override
		{
			return 0.f;
		}

	public:
		float etaB;
	};

	// microfacet reflection
	class MicrofacetReflection : public BxDF
	{
	public:
		CPU_GPU virtual Spectrum f(const BSDFData& data, const glm::vec3& wo, const glm::vec3& wi) const override
		{
			float cosThetaO = AbsCosTheta(wo);
			float cosThetaI = AbsCosTheta(wi);
			glm::vec3 wh = wi + wo;
			// Handle degenerate cases for microfacet reflection
			if (cosThetaI == 0 || cosThetaO == 0) return Spectrum();
			if (wh.x == 0 && wh.y == 0 && wh.z == 0) return Spectrum();
			wh = glm::normalize(wh);
			float F = 1.f;
			float D = TrowbridgeReitzD(wh, data.roughness);
			float G = TrowbridgeReitzG(wo, wi, data.roughness);

			return data.R * F * D * G / (4.f * cosThetaI * cosThetaO);
		}

		CPU_GPU virtual BSDFSample Sample_f(const BSDFData& data, const glm::vec3& wo, RNG& rng) const override
		{
			if (wo.z <= 0.f)
			{
				return BSDFSample();
			}
			float roughness = data.roughness;
			glm::vec3 wh = Sample_wh(wo, {rng.rand(), rng.rand()}, roughness);
			glm::vec3 wi = glm::reflect(-wo, wh);

			glm::vec3 wiW = glm::normalize(LocalToWorld(data.normal) * wi);
			
			if (!SameHemisphere(wo, wi)) return BSDFSample();

			// Compute PDF of _wi_ for microfacet reflection
			float pdf = TrowbridgeReitzPdf(wo, wh, roughness) / (4 * glm::dot(wo, wh));

			return BSDFSample(f(data, wo, wi), wiW, pdf, data.eta);
		}

		CPU_GPU virtual float PDF(const BSDFData& data, const glm::vec3& wo, const glm::vec3& wi) const override
		{
			glm::vec3 wh = glm::normalize(wo + wi);
			return TrowbridgeReitzPdf(wo, wh, data.roughness) / (4 * glm::dot(wo, wh));
		}
	};

	// matellic workflow
	class MetallicWorkflow : public BxDF 
	{
	public:
		CPU_GPU virtual Spectrum f(const BSDFData& data, const glm::vec3& wo, const glm::vec3& wi) const override
		{
			float cosThetaO = CosTheta(wo);
			float cosThetaI = CosTheta(wi);

			// Handle degenerate cases for microfacet reflection
			if (cosThetaI * cosThetaO < 1e-7f) return Spectrum(0.f);

						float roughness = glm::max(data.roughness, 0.01f);
			float metallic = data.metallic;
			float alpha = roughness * roughness;

			glm::vec3 wh = glm::normalize(wo + wi);

			Spectrum F = FrSchlick(glm::mix(Spectrum(0.04f), data.R, metallic), glm::dot(wh, wo));
			float D = SmithG(cosThetaO, cosThetaI, alpha);
			float G = GTR2Distrib(CosTheta(wh), alpha);

			return glm::mix(data.R * InvPi * (1.f - metallic), glm::vec3(G * D / (4.f * cosThetaO * cosThetaI)), F);
			// Equal to
			// (1.f - F0) * (1.f - metallic) * diffuse +  F0 * specular;
		}

		CPU_GPU virtual BSDFSample Sample_f(const BSDFData& data, const glm::vec3& wo, RNG& rng) const override
		{
			float roughness = glm::max(data.roughness, 0.01f);
			const float& metallic = data.metallic;
			float alpha = roughness * roughness;
			glm::vec3 wi, wiW;
			if (rng.rand() > (1.f / (2.f - metallic)))
			{
				wi = Sampler::SquareToHemisphereCosine({ rng.rand(), rng.rand() });
			}
			else
			{
				glm::vec3 wh = glm::normalize(GTR2Sample(data.normal, wo, alpha, { rng.rand(), rng.rand() }));
				wi = glm::reflect(-wo, wh);
			}

			wiW = glm::normalize(LocalToWorld(data.normal) * wi);
			if (wi.z < 0.f)
			{
				return BSDFSample();
			}
			else
			{
				return BSDFSample(f(data, wo, wi), wiW, PDF(data, wo, wi), data.eta);
			}
		}

		CPU_GPU virtual float PDF(const BSDFData& data, const glm::vec3& wo, const glm::vec3& wi) const override
		{
			glm::vec3 wh = glm::normalize(wo + wi);
			
			float roughness = glm::max(data.roughness, 0.01f);

			return glm::mix(CosTheta(wi) * InvPi,
						    GTR2Pdf(data.normal, wh, wo, roughness * roughness) / (4.f * AbsDot(wh, wo)),
						    1.f / (2.f - data.metallic));
		}
	};
}
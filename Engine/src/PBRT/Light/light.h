#pragma once

#include "PBRT/pbrtDefine.h"
#include "PBRT/spectrum.h"
#include "PBRT/intersection.h"
#include "PBRT/ray.h"

#include "PBRT/Shape/shape.h"

namespace CudaPBRT
{
	CPU_GPU Shape* Create(const ShapeData& data);

	enum class LightType
	{
		None = 0,
		ShapeLight,
		PointLight,
		SpotLight
	};

	struct LightSample 
	{
		Spectrum Le;
		glm::vec3 wiW;
		float pdf;

		Ray shadowRay;
	};

	struct LightData
	{
		LightType type;
		ShapeData shapeData;
		Spectrum Le;

		LightData(LightType type, const ShapeData& shapeData, const Spectrum& Le)
			: type(type), shapeData(shapeData), Le(Le)
		{}
	};

	class Light
	{
	public:
		CPU_GPU virtual ~Light() {}
		CPU_GPU virtual Spectrum GetLe() = 0;
		CPU_GPU virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const { return false; }

		CPU_GPU virtual LightSample Sample_Li(const glm::vec3& p, const glm::vec3& normal, const glm::vec2& xi) const = 0;
	};

	class ShapeLight : public Light {
	public:
		// AreaLight Interface
		CPU_GPU ShapeLight(const LightData& data)
			: Le(data.Le)
		{
			m_Shape = Create(data.shapeData);
		}
		CPU_GPU ~ShapeLight()
		{
			delete m_Shape;
		}

		CPU_GPU virtual Spectrum GetLe() override
		{
			return Le;
		}

		CPU_GPU virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const
		{
			return m_Shape->IntersectionP(ray, intersection);
		}
	
		CPU_GPU virtual LightSample Sample_Li(const glm::vec3& p, const glm::vec3& normal, const glm::vec2& xi) const override
		{
			LightSample sample;

			glm::vec3 sampled_point = m_Shape->Sample(xi);
			float area = m_Shape->Area();

			// compute r, r*r and cosTheta
			glm::vec3 r = sampled_point - p;

			glm::vec3 p_normal = glm::vec3(0, 0, 1);
			p_normal = m_Shape->m_TransformInvTranspose * p_normal;

			// set wiW and pdf
			sample.wiW = normalize(r);
			float cosTheta = glm::dot(-sample.wiW, p_normal);
			sample.pdf = glm::dot(r, r) / (cosTheta * area);

			sample.Le = Le;

			// spawn ray & check whether it can reach the light
			sample.shadowRay = Ray::SpawnRay(p, sample.wiW);

			return sample;
		}

	protected:
		Spectrum Le;
		Shape* m_Shape;
	};
}
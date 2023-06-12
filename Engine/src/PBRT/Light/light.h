#pragma once

#include "PBRT/pbrtDefine.h"
#include "PBRT/pbrtUtilities.h"

#include "PBRT/spectrum.h"
#include "PBRT/intersection.h"
#include "PBRT/ray.h"

#include "PBRT/Shape/shape.h"

namespace CudaPBRT
{
	CPU_GPU Shape* Create(const ShapeData& data);

	enum class LightType : unsigned char
	{
		None = 0,
		ShapeLight,
		PointLight,
		SpotLight
	};
	
	LightType Str2LightType(const char* str);

	struct LightSample 
	{
		Spectrum Le = Spectrum(0.f);
		glm::vec3 wiW = glm::vec3(0.f);
		float pdf = -1.f;
		float t = -1.f;
		Ray shadowRay = Ray();
		
		CPU_GPU LightSample()
		{}

		CPU_GPU LightSample(const Spectrum& Le, const glm::vec3& wiW, float pdf, float t, const Ray& ray)
			: Le(Le), wiW(wiW), pdf(pdf), t(t), shadowRay(ray)
		{}
	};

	struct LightData
	{
		LightType type;
		Shape** shapes;
		int shapeId;
		bool doubleSide;
		Spectrum Le;

		LightData(LightType type, Shape** shapes, int shape_id, const Spectrum& Le, bool doubleSide = false)
			: type(type), shapes(shapes), shapeId(shape_id), doubleSide(doubleSide), Le(Le)
		{}

		LightData(const LightData& other, Shape** shapes)
			: type(other.type), shapes(shapes), shapeId(other.shapeId), doubleSide(other.doubleSide), Le(other.Le)
		{}
	};

	class Light
	{
	public:
		CPU_GPU virtual ~Light() {}
		CPU_GPU virtual Spectrum GetLe() = 0;
		CPU_GPU virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const { return false; }
		CPU_GPU virtual int GetShapeId() const { return -1; }

		CPU_GPU virtual LightSample Sample_Li(const glm::vec3& p, const glm::vec3& normal, const glm::vec2& xi) const = 0;
		CPU_GPU virtual float PDF(const glm::vec3& p, const glm::vec3& wiW, float t, const glm::vec3& normal) const = 0;
	};

	class ShapeLight : public Light 
	{
	public:
		// AreaLight Interface
		CPU_GPU ShapeLight(const LightData& data)
			: Le(data.Le), m_ShapeId(data.shapeId), m_Shapes(data.shapes), m_DoubleSide(data.doubleSide)
		{
		}

		CPU_GPU virtual Spectrum GetLe() override
		{
			return Le;
		}

		CPU_GPU virtual int GetShapeId() const override { return m_ShapeId; }

		CPU_GPU virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const
		{
			return m_Shapes[m_ShapeId]->IntersectionP(ray, intersection);
		}
	
		CPU_GPU virtual LightSample Sample_Li(const glm::vec3& p, const glm::vec3& normal, const glm::vec2& xi) const override
		{
			glm::vec3 sampled_point = m_Shapes[m_ShapeId]->Sample(xi);

			// compute r, wiW
			glm::vec3 r = sampled_point - p;
			
			glm::vec3 wiW = glm::normalize(r);
			float t = glm::length(r);

			return { Le , wiW, ComputePDF(p, wiW, t), t, Ray::SpawnRay(p, wiW) };
		}

		CPU_GPU virtual float PDF(const glm::vec3& p, const glm::vec3& wiW, float t, const glm::vec3& normal) const override
		{
			return ComputePDF(p, wiW, t);
		}
	protected:
		INLINE CPU_GPU float ComputePDF(const glm::vec3& p, const glm::vec3& wiW, float t) const
		{
			float area = m_Shapes[m_ShapeId]->Area();
			
			glm::vec3 p_normal = m_Shapes[m_ShapeId]->GetNormal(p);

			float cosTheta = glm::dot(-wiW, p_normal);
			if (m_DoubleSide)
			{
				cosTheta = glm::abs(cosTheta);
			}
			return (t * t / (cosTheta * area));
		}
	protected:
		Spectrum Le;
		int m_ShapeId;
		Shape** m_Shapes;
		bool m_DoubleSide;
	};
}
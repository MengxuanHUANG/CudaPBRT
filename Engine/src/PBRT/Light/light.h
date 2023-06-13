#pragma once

#include "PBRT/pbrtDefine.h"
#include "PBRT/pbrtUtilities.h"

#include "PBRT/spectrum.h"
#include "PBRT/intersection.h"
#include "PBRT/ray.h"

#include "PBRT/Shape/shape.h"
#include "PBRT/Material/material.h"

namespace CudaPBRT
{
	class Light;

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
		const Light* light;
		glm::vec3 wiW = glm::vec3(0.f);
		float pdf = -1.f;
		float t = -1.f;
		Ray shadowRay = Ray();
		
		CPU_GPU LightSample()
		{}

		CPU_GPU LightSample(const Light* light, const glm::vec3& wiW, float pdf, float t, const Ray& ray)
			: light(light), wiW(wiW), pdf(pdf), t(t), shadowRay(ray)
		{}
	};

	struct LightData
	{
		LightType type;
		Shape** shapes;
		Material** materials;
		int shapeId;
		bool doubleSide;
		Spectrum irradiance;

		LightData(LightType type, Shape** shapes, Material** materials, int shape_id, const Spectrum& irradiance, bool doubleSide = false)
			: type(type), shapes(shapes), materials(materials), shapeId(shape_id), doubleSide(doubleSide), irradiance(irradiance)
		{}

		LightData(const LightData& other, Shape** shapes, Material** materials)
			: type(other.type), 
			  shapes(shapes), materials(materials), 
			  shapeId(other.shapeId), doubleSide(other.doubleSide), irradiance(other.irradiance)
		{}
	};

	class Light
	{
	public:
		CPU_GPU virtual ~Light() {}
		GPU_ONLY virtual Spectrum GetLe(const glm::vec3& p = glm::vec3(0.f)) const = 0;
		CPU_GPU virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const { return false; }
		CPU_GPU virtual int GetShapeId() const { return -1; }

		GPU_ONLY virtual LightSample Sample_Li(const glm::vec3& p, const glm::vec3& normal, const glm::vec2& xi) const = 0;
		GPU_ONLY virtual float PDF(const glm::vec3& p, const glm::vec3& wiW, float t, const glm::vec3& normal) const = 0;
	};

	class ShapeLight : public Light 
	{
	public:
		// AreaLight Interface
		CPU_GPU ShapeLight(const LightData& data)
			: m_Materials(data.materials), m_ShapeId(data.shapeId), m_Shapes(data.shapes), m_DoubleSide(data.doubleSide)
		{
		}

		GPU_ONLY virtual Spectrum GetLe(const glm::vec3& p = glm::vec3(0.f)) const override
		{
			return m_Materials[m_Shapes[m_ShapeId]->material_id]->GetIrradiance(m_Shapes[m_ShapeId]->GetUV(p));
		}

		CPU_GPU virtual int GetShapeId() const override { return m_ShapeId; }

		CPU_GPU virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const
		{
			return m_Shapes[m_ShapeId]->IntersectionP(ray, intersection);
		}
	
		GPU_ONLY virtual LightSample Sample_Li(const glm::vec3& p, const glm::vec3& normal, const glm::vec2& xi) const override
		{
			glm::vec3 sampled_point = m_Shapes[m_ShapeId]->Sample(xi);

			// compute r, wiW
			glm::vec3 r = sampled_point - p;
			
			glm::vec3 wiW = glm::normalize(r);
			float t = glm::length(r);

			return { this,
					 wiW, 
					 ComputePDF(sampled_point, wiW, t),
					 t, 
					 Ray::SpawnRay(p, wiW)};
		}

		GPU_ONLY virtual float PDF(const glm::vec3& p, const glm::vec3& wiW, float t, const glm::vec3& normal) const override
		{
			return ComputePDF(p, wiW, t);
		}
	protected:
		INLINE GPU_ONLY float ComputePDF(const glm::vec3& p, const glm::vec3& wiW, float t) const
		{
			float area = m_Shapes[m_ShapeId]->Area();
			
			glm::vec3 p_normal = m_Shapes[m_ShapeId]->GetNormal(p);
			// apply normal map
			p_normal = m_Materials[m_Shapes[m_ShapeId]->material_id]->GetNormal(p_normal, m_Shapes[m_ShapeId]->GetUV(p));

			float cosTheta = glm::dot(-wiW, p_normal);
			if (m_DoubleSide)
			{
				cosTheta = glm::abs(cosTheta);
			}
			return (t * t / (cosTheta * area));
		}
	public:
		int m_ShapeId;
		Shape** m_Shapes;
		Material** m_Materials;
		bool m_DoubleSide;
	};
}
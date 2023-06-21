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

	CPU_GPU void Create(Shape* shape_ptr, const ShapeData& data);
	CPU_GPU void Create(Material* material_ptr, const MaterialData& data);

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
		float pdf = -1.f;
		glm::vec3 p = glm::vec3(0.f);
		glm::vec3 wiW = glm::vec3(0.f);
		
		CPU_GPU LightSample()
			: light(nullptr), pdf(-1.f), p(0.f), wiW(0.f)
		{}

		CPU_GPU LightSample(const LightSample& other)
			: light(other.light), pdf(other.pdf), p(other.p), wiW(other.wiW)
		{}

		CPU_GPU LightSample(const Light* light, float pdf, const glm::vec3& p, const glm::vec3& wiW)
			: light(light), pdf(pdf), p(p), wiW(wiW)
		{}

		CPU_GPU LightSample& operator=(const LightSample& other)
		{
			light = other.light;
			pdf = other.pdf;
			p = other.p;
			wiW = other.wiW;
			return *this;
		}
	};

	struct LightData
	{
		LightType type;
		ShapeData shape_data;
		MaterialData material_data;
		int shapeId;
		bool doubleSide;
		Spectrum irradiance;

		LightData(LightType type, const ShapeData& shape_data,  const MaterialData& material_data, int shape_id, const Spectrum& irradiance, bool doubleSide = false)
			: type(type), shape_data(shape_data), material_data(material_data), shapeId(shape_id), doubleSide(doubleSide), irradiance(irradiance)
		{}
	};

	struct ShapeLightUnionData
	{
		int shapeId;
		Shape shape;
		Material material;
		bool doubleSide;
	};

	struct PointLightUnionData
	{
		Spectrum irradiance = Spectrum(0.f);
	};

	union UnionLightData
	{
		ShapeLightUnionData shapeLightData;
		PointLightUnionData pointLightData;
	};

	class Light
	{
	public:
		CPU_GPU Light() {}
		GPU_ONLY virtual Spectrum GetLe(const glm::vec3& p = glm::vec3(0.f)) const = 0;
		CPU_GPU virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const { return false; }
		CPU_GPU virtual int GetShapeId() const { return -1; }

		GPU_ONLY virtual LightSample Sample_Li(const glm::vec3& p, const glm::vec3& normal, const glm::vec2& xi) const = 0;
		GPU_ONLY virtual float PDF(const glm::vec3& p, const glm::vec3& wiW, float t, const glm::vec3& normal) const = 0;
	
	public:
		ShapeLightUnionData shapeLightData;
	};

	class ShapeLight : public Light 
	{
#define m_ShapeId shapeLightData.shapeId
#define m_Shape shapeLightData.shape
#define m_Material shapeLightData.material
#define m_DoubleSide shapeLightData.doubleSide
	public:
		// AreaLight Interface
		CPU_GPU ShapeLight(const LightData& data)
		{
			m_ShapeId = data.shapeId;
			Create(&m_Shape, data.shape_data);
			Create(&m_Material, data.material_data);

			m_DoubleSide = data.doubleSide;
		}

		GPU_ONLY virtual Spectrum GetLe(const glm::vec3& p = glm::vec3(0.f)) const override
		{
			return m_Material.GetIrradiance(p, m_Shape);
		}

		CPU_GPU virtual int GetShapeId() const override { return m_ShapeId; }

		CPU_GPU virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const
		{
			return m_Shape.IntersectionP(ray, intersection);
		}
	
		GPU_ONLY virtual LightSample Sample_Li(const glm::vec3& p, const glm::vec3& normal, const glm::vec2& xi) const override
		{
			glm::vec3 sampled_point = m_Shape.Sample(xi);

			// compute r, wiW
			glm::vec3 r = sampled_point - p;
			
			glm::vec3 wiW = glm::normalize(r);
			float t2 = glm::dot(r, r);

			return { this,
					 ComputePDF(sampled_point, wiW, t2),
					 sampled_point,
				     wiW};
		}

		GPU_ONLY virtual float PDF(const glm::vec3& p, const glm::vec3& wiW, float t, const glm::vec3& normal) const override
		{
			return ComputePDF(p, wiW, t * t);
		}
	protected:
		INLINE GPU_ONLY float ComputePDF(const glm::vec3& p, const glm::vec3& wiW, float t2) const
		{
			glm::vec3 p_normal = m_Shape.GetNormal(p);

			float cosTheta = (m_DoubleSide ? AbsDot(-wiW, p_normal) : glm::dot(-wiW, p_normal));

			return (t2 / (cosTheta * m_Shape.Area()));
		}

#undef m_ShapeId
#undef m_Shape
#undef m_Material
#undef m_DoubleSide
	};
}
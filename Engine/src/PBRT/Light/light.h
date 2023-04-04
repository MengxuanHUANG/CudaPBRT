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
	
	protected:
		Spectrum Le;
		Shape* m_Shape;
	};
}
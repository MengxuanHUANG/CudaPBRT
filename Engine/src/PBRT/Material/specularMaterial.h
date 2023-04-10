#pragma once
#include "PBRT/pbrtDefine.h"
#include "material.h"
#include "PBRT/spectrum.h"
#include "PBRT/bsdf.h"

namespace CudaPBRT
{
	// GPU side object
	class SpecularMaterial : public Material
	{
	public:
		CPU_GPU SpecularMaterial(const MaterialData& mData)
			:Material(mData), m_BSDF(new SpecularReflection())
		{
		}

		CPU_GPU ~SpecularMaterial() {}

		CPU_GPU virtual BSDF& GetBSDF() override
		{
			return m_BSDF;
		}

	protected:
		BSDF m_BSDF;
	};
}

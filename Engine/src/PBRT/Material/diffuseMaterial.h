#pragma once
#include "PBRT/pbrtDefine.h"
#include "material.h"
#include "PBRT/spectrum.h"
#include "PBRT/bsdf.h"

namespace CudaPBRT
{
	// GPU side object
	class DiffuseMaterial : public Material
	{
	public:
		CPU_GPU DiffuseMaterial(const MaterialData& mData)
			:Material(mData), m_BSDF(new LambertianReflection(m_MaterialData.eta))
		{
		}

		CPU_GPU ~DiffuseMaterial() {}

		CPU_GPU virtual BSDF& GetBSDF() override
		{
			return m_BSDF;
		}

	protected:
		BSDF m_BSDF;
	};
}

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
			:Material(mData)
		{}

		CPU_GPU ~DiffuseMaterial() {}

		CPU_GPU virtual BSDF GetBSDF() const override
		{
			return BSDF(new LambertianReflection(Spectrum(GetAlbedo()), m_MaterialData.eta));
		}
	};
}

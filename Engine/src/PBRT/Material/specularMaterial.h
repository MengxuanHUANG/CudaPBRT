#pragma once
#include "PBRT/pbrtDefine.h"
#include "material.h"

namespace CudaPBRT
{
	// GPU side object
	class SpecularMaterial : public Material
	{
	public:
		CPU_GPU SpecularMaterial(const MaterialData& mData)
			:Material(mData, new SpecularReflection())
		{
		}
	};
}

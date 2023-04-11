#pragma once
#include "PBRT/pbrtDefine.h"
#include "material.h"

namespace CudaPBRT
{
	// GPU side object
	class DiffuseMaterial : public Material
	{
	public:
		CPU_GPU DiffuseMaterial(const MaterialData& mData)
			:Material(mData, new LambertianReflection())
		{
		}
	};
}

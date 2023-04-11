#pragma once
#include "PBRT/pbrtDefine.h"
#include "material.h"

namespace CudaPBRT
{
	// GPU side object
	class Glass : public Material
	{
	public:
		CPU_GPU Glass(const MaterialData& mData)
			:Material(mData, new GlassBxDF(mData.eta))
		{
		}
	};
}
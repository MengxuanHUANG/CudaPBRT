#pragma once

#include "pbrtDefine.h"

#include "spectrum.h"
#include "ray.h"
#include "intersection.h"

namespace CudaPBRT
{
	struct PathSegment
	{
		 int depth = 0;
		 int pixelId = -1;
		 Spectrum throughput = Spectrum(1.f);
		 Spectrum radiance = Spectrum(0.f);

		 Ray ray = Ray();
		 Intersection intersection;

		 float eta = AirETA;

		 INLINE CPU_GPU bool operator<(const PathSegment& other) const
		 {
			 return intersection.material_id < other.intersection.material_id;
		 }

		 INLINE CPU_GPU void Reset()
		 {
			 depth = 0;
			 pixelId = -1;
			 throughput = Spectrum(1.f);
			 radiance = Spectrum(0.f);
			 ray = Ray();
			 intersection = Intersection();
			 eta = AirETA;
		 }

		 INLINE CPU_GPU void End()
		 {
			 depth = CudaPBRT::PathMaxDepth;
		 }

		 INLINE CPU_GPU bool IsEnd() const { return depth >= CudaPBRT::PathMaxDepth;  }
	};
}
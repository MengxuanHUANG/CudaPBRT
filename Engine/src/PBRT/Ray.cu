#include "Ray.cuh"

#include "pbrt.h"

namespace CudaPBRT
{
	Ray Ray::SpawnRay(const glm::vec3& o, const glm::vec3& dir)
	{
		return {o + dir * gamma(3), dir};
	}
}
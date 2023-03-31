#pragma once

#include "pbrtDefine.h"

#include <glm/glm.hpp>

namespace CudaPBRT
{
	class Ray
	{
	public:
		GPU_ONLY Ray(const glm::vec3& o, const glm::vec3& dir)
			:O(o), DIR(dir)
		{
		}

		GPU_ONLY glm::vec3 operator*(const float& t) const
		{
			return O + t * DIR;
		}

	public:
		float tMax;
		glm::vec3 O, DIR;

	public:
		GPU_ONLY static Ray SpawnRay(const glm::vec3& o, const glm::vec3& dir)
		{
			return { o + dir * gamma(3), dir };
		}
	};
}
#pragma once

#include "pbrtDefine.h"

#include <glm/glm.hpp>

namespace CudaPBRT
{
	class Ray
	{
	public:
		CPU_GPU Ray()
			:O(0.f), DIR(0.f)
		{}

		CPU_GPU Ray(const glm::vec3& o, const glm::vec3& dir)
			:O(o), DIR(dir)
		{
		}

		INLINE CPU_GPU glm::vec3 operator*(const float& t) const
		{
			return O + t * DIR;
		}

	public:
		float tMax;
		glm::vec3 O, DIR;

	public:
		INLINE CPU_GPU static Ray SpawnRay(const glm::vec3& o, const glm::vec3& dir)
		{
			return { o + dir * 0.0001f, dir };
		}
	};
}
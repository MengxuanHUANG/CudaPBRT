#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

namespace CudaPBRT
{
	class Ray
	{
	public:
		__host__ __device__ Ray(const glm::vec3& o, const glm::vec3& dir)
			:O(o), DIR(dir)
		{
		}

		__device__ glm::vec3 operator*(float t)
		{
			return O + t * DIR;
		}

	public:
		float tMax;
		glm::vec3 O, DIR;

	public:
		static Ray SpawnRay(const glm::vec3& o, const glm::vec3& dir);
	};
}
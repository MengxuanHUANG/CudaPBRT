#pragma once

#include <glm/glm.hpp>

namespace CudaPBRT
{
	class Ray
	{
	public:
		Ray(const glm::vec3& o, const glm::vec3& dir);
		
		glm::vec3 operator*(float t);

	public:
		float tMax;
		glm::vec3 O, DIR;

	public:
		static Ray SpawnRay(const glm::vec3& o, const glm::vec3& dir);
	};
}
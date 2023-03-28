#include "Ray.h"

namespace CudaPBRT
{
	Ray::Ray(const glm::vec3& o, const glm::vec3& dir)
		:O(o), DIR(dir)
	{}

	glm::vec3 Ray::operator*(float t)
	{
		return O + t * DIR;
	}

	Ray Ray::SpawnRay(const glm::vec3& o, const glm::vec3& dir)
	{
		return { o + 0.01f * dir, dir };
	}
}
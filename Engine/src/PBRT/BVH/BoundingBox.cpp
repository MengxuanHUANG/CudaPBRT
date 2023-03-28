#include "BoundingBox.h"

#include "PBRT/pbrt.h"

namespace CudaPBRT
{
	BoundingBox::BoundingBox()
		: m_Min(glm::vec3(std::numeric_limits<float>::min())),
		  m_Max(glm::vec3(std::numeric_limits<float>::max()))
	{}

	BoundingBox::BoundingBox(const glm::vec3& pMin, const glm::vec3& pMax)
		: m_Min(pMin), m_Max(pMax)
	{}

	bool BoundingBox::IntersectP(const Ray& ray, float* hit_t0, float* hit_t1) const
	{
		float t0 = 0, t1 = ray.tMax;

		for (int i = 0; i < 3; ++i) // check x, y, z
		{
			float invRayDir = 1.f / ray.DIR[i];

			float tNear = (m_Min[i] - ray.O[i]) * invRayDir;
			float tFar  = (m_Max[i] - ray.O[i]) * invRayDir;

			// swap near and far based on ray's direction
			if (tNear > tFar) std::swap(tNear, tFar);

			tFar *= 1 + 2 * gamma(3);
			t0 = glm::max(tNear, t0);
			t1 = glm::min(tFar, t1);

			if (t0 > t1) return false;
		}

		if (hit_t0) *hit_t0 = t0;
		if (hit_t1) *hit_t1 = t1;

		return true;
	}
}
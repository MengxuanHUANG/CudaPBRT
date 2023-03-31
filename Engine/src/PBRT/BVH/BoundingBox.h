#pragma once
#include "Core/Core.h"
#include "PBRT/pbrtDefine.h"
#include "PBRT/ray.h"
#include <glm/glm.hpp>


namespace CudaPBRT
{
	class BoundingBox
	{
	public:
		GPU_ONLY BoundingBox()
			: m_Min(FloatMin),
			  m_Max(FloatMax)
		{}
		GPU_ONLY BoundingBox(const glm::vec3& pMin, const glm::vec3& pMax)
			: m_Min(pMin), m_Max(pMax)
		{}

		GPU_ONLY glm::vec3& operator[](int i)
		{
			ASSERT(i == 0 || i == 1);
			return (i == 0 ? m_Min : m_Max);
		}

		GPU_ONLY inline glm::vec3 Diagonal() const { return m_Max - m_Min; }

		GPU_ONLY inline glm::vec3 Corner(int corner)
		{
			return glm::vec3((*this)[(corner & 1)].x,
							 (*this)[(corner & 2) ? 1 : 0].y,
							 (*this)[(corner & 4) ? 1 : 0].z);
		}

		GPU_ONLY inline BoundingBox Union(const BoundingBox& bBox) const
		{
			return { glm::vec3(
						glm::min(m_Min.x, bBox.m_Min.x),
						glm::min(m_Min.y, bBox.m_Min.y),
						glm::min(m_Min.z, bBox.m_Min.z)
					),
					glm::vec3(
						glm::max(m_Max.x, bBox.m_Max.x),
						glm::max(m_Max.y, bBox.m_Max.y),
						glm::max(m_Max.z, bBox.m_Max.z)
					)};
		}

		GPU_ONLY inline BoundingBox Intersect(const BoundingBox& bBox) const
		{
			return { glm::vec3(
						glm::max(m_Min.x, bBox.m_Min.x),
						glm::max(m_Min.y, bBox.m_Min.y),
						glm::max(m_Min.z, bBox.m_Min.z)
					),
					glm::vec3(
						glm::min(m_Max.x, bBox.m_Max.x),
						glm::min(m_Max.y, bBox.m_Max.y),
						glm::min(m_Max.z, bBox.m_Max.z)
					) };
		}

		GPU_ONLY inline bool Overlap(const BoundingBox& bBox) const
		{
			bool x = (m_Max.x >= bBox.m_Min.x) && (m_Min.x <= bBox.m_Max.x);
			bool y = (m_Max.y >= bBox.m_Min.y) && (m_Min.y <= bBox.m_Max.y);
			bool z = (m_Max.z >= bBox.m_Min.z) && (m_Min.z <= bBox.m_Max.z);
			return (x && y && z);
		}

		GPU_ONLY inline bool Inside(const glm::vec3& p) const
		{
			return ((p.x >= m_Min.x) && (p.x <= m_Max.x) && 
					(p.y >= m_Min.y) && (p.y <= m_Max.y) &&
					(p.z >= m_Min.z) && (p.z <= m_Max.z));
		}

		GPU_ONLY inline bool InsideExclusive(const BoundingBox& bBox)
		{
			return (m_Min.x >= bBox.m_Min.x && m_Min.x < bBox.m_Max.x&&
				m_Min.y >= bBox.m_Min.y && m_Min.y < bBox.m_Max.y&&
				m_Min.z >= bBox.m_Min.z && m_Min.z < bBox.m_Max.z);
		}

		GPU_ONLY inline float SurfaceArea() const
		{
			glm::vec3 d = Diagonal();
			return 2.f * (d.x * d.y + d.x * d.z + d.y * d.z);
		}

		GPU_ONLY inline float Volume() const
		{
			glm::vec3 d = Diagonal();
			return d.x * d.y * d.z;
		}

		GPU_ONLY int MaximumExtent() const
		{
			glm::vec3 d = Diagonal();
			if (d.x > d.y && d.x > d.z) { return 0; }
			else if (d.y > d.z) { return 1; }
			else { return 2; }
		}

		GPU_ONLY inline glm::vec3 Lerp(const glm::vec3& t) const
		{
			return glm::mix(m_Min, m_Max, t);
		}

		GPU_ONLY inline glm::vec3 Offset(const glm::vec3 &p) const {
			glm::vec3 o = p - m_Min;
			if (m_Max.x > m_Min.x) o.x /= (m_Max.x - m_Min.x);
			if (m_Max.y > m_Min.y) o.y /= (m_Max.y - m_Min.y);
			if (m_Max.z > m_Min.z) o.z /= (m_Max.z - m_Min.z);

			return o;
		}

		GPU_ONLY inline void BoundingSphere(glm::vec3& center, float& radius) const {
			center = (m_Min + m_Max) / 2.f;
			radius = Inside(center) ? glm::length(center - m_Max) : 0.f;
		}

		GPU_ONLY int IntersectP(const Ray& ray, float* hit_t0, float* hit_t1) const
		{
			float t0 = 0, t1 = ray.tMax;
			float axis = -1;
			for (int i = 0; i < 3; ++i) // check x, y, z
			{
				float invRayDir = 1.f / ray.DIR[i];

				float tNear = (m_Min[i] - ray.O[i]) * invRayDir;
				float tFar = (m_Max[i] - ray.O[i]) * invRayDir;

				// swap near and far based on ray's direction
				if (tNear > tFar)
				{
					float temp = tNear;
					tNear = tFar;
					tFar = temp;
				}

				tFar *= 1 + 2 * gamma(3);
				if (tNear > t0)
				{
					axis = i;
					t0 = glm::max(tNear, t0);
				}
				t1 = glm::min(tFar, t1);

				if (t0 > t1) return -1;
			}

			if (hit_t0) *hit_t0 = t0;
			if (hit_t1) *hit_t1 = t1;

			return axis;
		}
	
protected:
		glm::vec3 m_Min, m_Max;
	};
}
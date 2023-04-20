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
		CPU_GPU BoundingBox()
			: m_Min(FloatMax, FloatMax, FloatMax),
			  m_Max(FloatMin, FloatMin, FloatMin)
		{}
		CPU_GPU BoundingBox(const glm::vec3& pMin, const glm::vec3& pMax)
			: m_Min(pMin), m_Max(pMax)
		{}

		CPU_GPU glm::vec3& operator[](int i)
		{
			ASSERT(i == 0 || i == 1);
			return (i == 0 ? m_Min : m_Max);
		}

		CPU_GPU inline glm::vec3 Centroid() const { return (m_Min + m_Max) * 0.5f; }

		CPU_GPU inline glm::vec3 Diagonal() const { return m_Max - m_Min; }

		CPU_GPU inline glm::vec3 Corner(int corner)
		{
			return glm::vec3((*this)[(corner & 1)].x,
							 (*this)[(corner & 2) ? 1 : 0].y,
							 (*this)[(corner & 4) ? 1 : 0].z);
		}

		CPU_GPU inline void Union(const BoundingBox& other)
		{
			m_Min.x = other.m_Min.x < m_Min.x ? other.m_Min.x : m_Min.x;
			m_Min.y = other.m_Min.y < m_Min.y ? other.m_Min.y : m_Min.y;
			m_Min.z = other.m_Min.z < m_Min.z ? other.m_Min.z : m_Min.z;

			m_Max.x = other.m_Max.x > m_Max.x ? other.m_Max.x : m_Max.x;
			m_Max.y = other.m_Max.y > m_Max.y ? other.m_Max.y : m_Max.y;
			m_Max.z = other.m_Max.z > m_Max.z ? other.m_Max.z : m_Max.z;
		}

		CPU_GPU inline void Union(const glm::vec3& p)
		{
			m_Min.x = p.x < m_Min.x ? p.x : m_Min.x;
			m_Min.y = p.y < m_Min.y ? p.y : m_Min.y;
			m_Min.z = p.z < m_Min.z ? p.z : m_Min.z;

			m_Max.x = p.x > m_Max.x ? p.x : m_Max.x;
			m_Max.y = p.y > m_Max.y ? p.y : m_Max.y;
			m_Max.z = p.z > m_Max.z ? p.z : m_Max.z;
		}

		CPU_GPU inline BoundingBox Intersect(const BoundingBox& other) const
		{
			return { glm::vec3(
						m_Min.x > other.m_Min.x ? m_Min.x : other.m_Min.x,
						m_Min.y > other.m_Min.y ? m_Min.y : other.m_Min.y,
						m_Min.z > other.m_Min.z ? m_Min.z : other.m_Min.z
					),
					glm::vec3(
						m_Max.x < other.m_Max.x ? m_Max.x : other.m_Max.x,
						m_Max.y < other.m_Max.y ? m_Max.y : other.m_Max.y,
						m_Max.z < other.m_Max.z ? m_Max.z : other.m_Max.z
					) };
		}

		CPU_GPU inline bool Overlap(const BoundingBox& bBox) const
		{
			bool x = (m_Max.x >= bBox.m_Min.x) && (m_Min.x <= bBox.m_Max.x);
			bool y = (m_Max.y >= bBox.m_Min.y) && (m_Min.y <= bBox.m_Max.y);
			bool z = (m_Max.z >= bBox.m_Min.z) && (m_Min.z <= bBox.m_Max.z);
			return (x && y && z);
		}

		CPU_GPU inline bool Inside(const glm::vec3& p) const
		{
			return ((p.x >= m_Min.x) && (p.x <= m_Max.x) && 
					(p.y >= m_Min.y) && (p.y <= m_Max.y) &&
					(p.z >= m_Min.z) && (p.z <= m_Max.z));
		}

		CPU_GPU inline bool InsideExclusive(const BoundingBox& bBox)
		{
			return (m_Min.x >= bBox.m_Min.x && m_Min.x < bBox.m_Max.x&&
				m_Min.y >= bBox.m_Min.y && m_Min.y < bBox.m_Max.y&&
				m_Min.z >= bBox.m_Min.z && m_Min.z < bBox.m_Max.z);
		}

		CPU_GPU inline float SurfaceArea() const
		{
			glm::vec3 d = Diagonal();
			return 2.f * (d.x * d.y + d.x * d.z + d.y * d.z);
		}

		CPU_GPU inline float Volume() const
		{
			glm::vec3 d = Diagonal();
			return d.x * d.y * d.z;
		}

		CPU_GPU int MaximumExtent() const
		{
			glm::vec3 d = Diagonal();
			if (d.x > d.y && d.x > d.z) { return 0; }
			else if (d.y > d.z) { return 1; }
			else { return 2; }
		}

		CPU_GPU inline glm::vec3 Lerp(const glm::vec3& t) const
		{
			return glm::mix(m_Min, m_Max, t);
		}

		CPU_GPU inline glm::vec3 Offset(const glm::vec3 &p) const {
			glm::vec3 o = (p - m_Min) / (m_Max - m_Min);
			//if (m_Max.x > m_Min.x) o.x /= (m_Max.x - m_Min.x);
			//if (m_Max.y > m_Min.y) o.y /= (m_Max.y - m_Min.y);
			//if (m_Max.z > m_Min.z) o.z /= (m_Max.z - m_Min.z);

			return o;
		}

		CPU_GPU inline void BoundingSphere(glm::vec3& center, float& radius) const {
			center = (m_Min + m_Max) / 2.f;
			radius = Inside(center) ? glm::length(center - m_Max) : 0.f;
		}

		CPU_GPU bool IntersectP(const Ray& ray, const glm::vec3& inv_dir, float& t) const
		{
			glm::vec3 tNear = (m_Min - ray.O) * inv_dir;
			glm::vec3 tFar = (m_Max - ray.O) * inv_dir;

			glm::vec3 tmin = glm::min(tNear, tFar);
			glm::vec3 tmax = glm::max(tNear, tFar);

			float t0 = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
			float t1 = glm::min(glm::min(tmax.x, tmax.y), tmax.z);

			if (t0 > t1) return false;

			t = t0;

			return true;
		}
	
	public:
		glm::vec3 m_Min;
		glm::vec3 m_Max;
	};
}
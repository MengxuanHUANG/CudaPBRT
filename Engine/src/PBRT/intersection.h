#pragma once
#include "pbrtDefine.h"

#include <glm/glm.hpp>

namespace CudaPBRT
{
	struct Intersection
	{
		int id = -1;;
		int material_id = -1;
		bool isLight = false;

		float t = CudaPBRT::FloatMax;
		glm::vec3 p = glm::vec3(0.f);
		glm::vec3 normal = glm::vec3(0.f);

		glm::vec2 uv = glm::vec2(0.f);

		CPU_GPU Intersection() 
		{}

		CPU_GPU Intersection(const float& t,
							 const glm::vec3& p, 
							 const glm::vec3& n)
			:t(t), p(p), normal(n)
		{}

		INLINE CPU_GPU void Reset()
		{
			id = -1;;
			material_id = -1;
			isLight = false;

			t = CudaPBRT::FloatMax;
			p = glm::vec3(0.f);
			normal = glm::vec3(0.f);
			uv = glm::vec2(0.f);
		}

		INLINE CPU_GPU bool operator<(const Intersection& other) const
		{
			return t < other.t;
		}

		INLINE CPU_GPU bool operator<=(const Intersection & other) const
		{
			return t <= other.t;
		}

		INLINE CPU_GPU bool operator>(const Intersection& other) const
		{
			return t > other.t;
		}

		INLINE CPU_GPU bool operator>=(const Intersection& other) const
		{
			return t >= other.t;
		}
	};
}
#pragma once
#include "pbrtDefine.h"

#include <glm/glm.hpp>

namespace CudaPBRT
{
	struct Intersection
	{
		int id = -1;;
		float t = -1.f;
		glm::vec3 p = glm::vec3(0.f);
		glm::vec3 normal = glm::vec3(0.f);
		glm::vec3 wo = glm::vec3(0.f);

		GPU_ONLY Intersection() = default;

		GPU_ONLY Intersection(const float& t, 
							  const glm::vec3& p, 
							  const glm::vec3& n)
			:t(t), p(p), normal(n)
		{}
	};
}
#pragma once

#include <glm/glm.hpp>

namespace CudaPBRT
{
	struct Transform
	{
		glm::vec3 translation;
		glm::vec3 rotation;
		glm::vec3 scale;

		Transform() = default;
	};
}
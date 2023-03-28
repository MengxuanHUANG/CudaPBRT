#pragma once

#include <glm/glm.hpp>

namespace CudaPBRT
{
	struct MouseProps
	{
		bool button[5]{ false, false, false, false, false };
		glm::vec2 position;

		MouseProps(const glm::vec2& position)
			: position(position)
		{}
	};
}
#pragma once
#include "Core/Core.h"

namespace CudaPBRT
{
	class GraphicsContext
	{
	public:
		virtual void Init() = 0;
		virtual void SwapBuffers() = 0;

		static uPtr<GraphicsContext> CreateGraphicsContext(void* window);
	};
}
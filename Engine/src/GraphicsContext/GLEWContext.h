#pragma once

#include "Core/Core.h"
#include "GraphicsContext.h"

#include <GL/glew.h>
#include <glfw/glfw3.h>

namespace CudaPBRT
{
	class GLEWContext : public GraphicsContext
	{
	public:
		GLEWContext(void* window);

		virtual void Init() override;
		virtual void SwapBuffers() override;

	private:
		GLFWwindow* m_WindowHandle;
	};
}
#pragma once
#include "Window.h"
#include <GraphicsContext/GLEWContext.h>

#include <GL/glew.h>
#include <glfw/glfw3.h>

#include "Events/Event.h"

namespace CudaPBRT
{
	class WindowsWindow : public Window
	{
	public:
		WindowsWindow(EventCallbackFn fn, const WindowProps& props);
		virtual ~WindowsWindow();

		virtual void OnUpdate() override;

		// Be careful to reinterpret_cast the returned pointer
		virtual void* GetNativeWindow() const override;

	private:
		virtual void Init();

	public:
		void OnWindowResize(GLFWwindow* window, int width, int height);
		void OnWindowClose(GLFWwindow* window);

	protected:
		GLFWwindow* m_NativeWindow;
		
		struct WindowData
		{
			EventCallbackFn eventCallbackFn;
			WindowProps windowProps;

			WindowData(EventCallbackFn fn, const WindowProps& props)
				:eventCallbackFn(fn), windowProps(props)
			{}
		};

		WindowData m_WindowData;
		uPtr<GraphicsContext> m_GraphicsContext;
	};
}
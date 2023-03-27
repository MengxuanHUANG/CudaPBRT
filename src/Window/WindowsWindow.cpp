#include "WindowsWindow.h"

#include <iostream>
#include <format>
#include <string>

namespace CudaPBRT
{
	void errorCallback(int error, const char* description) {
		fprintf(stderr, "%s\n", description);
	}

	uPtr<Window> Window::CreateWindow(EventCallbackFn fn, const WindowProps& props)
	{
		return mkU<WindowsWindow>(fn, props);
	}

	WindowsWindow::WindowsWindow(EventCallbackFn fn, const WindowProps& props)
		: m_WindowData(fn, props)
	{
		this->Init();
	}

	WindowsWindow::~WindowsWindow()
	{
		glfwDestroyWindow(m_NativeWindow);
		glfwTerminate();
	}

	void WindowsWindow::Init()
	{
		glfwSetErrorCallback(errorCallback);

		if (!glfwInit()) {
			exit(EXIT_FAILURE);
		}

		m_NativeWindow = glfwCreateWindow(m_WindowData.windowProps.width, m_WindowData.windowProps.height, m_WindowData.windowProps.title.c_str(), NULL, NULL);
		if (!m_NativeWindow) {
			ASSERT(false);
			glfwTerminate();
		}

		glfwMakeContextCurrent(m_NativeWindow);

		m_GraphicsContext = GraphicsContext::CreateGraphicsContext(reinterpret_cast<void*>(m_NativeWindow));
		
		m_GraphicsContext->Init();

		int display_w, display_h;
		glfwGetFramebufferSize(m_NativeWindow, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);

		glfwSetWindowUserPointer(m_NativeWindow, &m_WindowData);

		// WindowResize Call back
		glfwSetWindowSizeCallback(m_NativeWindow, [](GLFWwindow* window, int width, int height) 
		{
			WindowResizeEvent event(width, height);
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

			data.windowProps.width = width;
			data.windowProps.height = height;

			data.eventCallbackFn(event);
		});
		glfwSetWindowCloseCallback(m_NativeWindow, [](GLFWwindow* window) 
		{
			WindowCloseEvent event;
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
			data.eventCallbackFn(event);
		});
		
		glfwSetMouseButtonCallback(m_NativeWindow, [](GLFWwindow* window, int button, int action, int mods)
		{
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

			switch (action)
			{
			case GLFW_PRESS:
			{
				MouseButtonPressedEvent event(button);
				data.eventCallbackFn(event);
				break;
			}
			case GLFW_RELEASE:
			{
				MouseButtonReleasedEvent event(button);
				data.eventCallbackFn(event);
				break;
			}
			}
		});

		glfwSetScrollCallback(m_NativeWindow, [](GLFWwindow* window, double X_Offset, double Y_Offset)
		{
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

			MouseScrolledEvent event((float)X_Offset, (float)Y_Offset);
			data.eventCallbackFn(event);
		});

		glfwSetCursorPosCallback(m_NativeWindow, [](GLFWwindow* window, double X_Pos, double Y_Pos)
		{
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

			MouseMovedEvent event((float)X_Pos, (float)Y_Pos);
			data.eventCallbackFn(event);
		});

		glfwSetKeyCallback(m_NativeWindow, [](GLFWwindow* window, int key, int scancode, int action, int mods)
		{
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

			switch (action)
			{
			case GLFW_PRESS:
			{
				KeyPressedEvent event(key, 0);
				data.eventCallbackFn(event);
				break;
			}
			case GLFW_RELEASE:
			{
				KeyReleasedEvent event(key);
				data.eventCallbackFn(event);
				break;
			}
			case GLFW_REPEAT:
			{
				KeyPressedEvent(key, 1);
				break;
			}
			}
		});

		glfwSetCharCallback(m_NativeWindow, [](GLFWwindow* window, unsigned int input_char)
		{
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

			KeyTypedEvent event(input_char);
			data.eventCallbackFn(event);
		});
	}

	void WindowsWindow::OnUpdate()
	{
		// Poll events
		glfwPollEvents();
		
		// Swap buffers
		m_GraphicsContext->SwapBuffers();
	}

	// Be careful to reinterpret_cast the returned pointer
	void* WindowsWindow::GetNativeWindow() const
	{
		return reinterpret_cast<void*>(m_NativeWindow);
	}

	void WindowsWindow::OnWindowResize(GLFWwindow* window, int width, int height)
	{
		std::string temp = R"(Window Resize: ({}, {}))";
		std::cout << std::vformat(temp, std::make_format_args(width, height)) << std::endl;
	}

	void WindowsWindow::OnWindowClose(GLFWwindow* window)
	{
		std::cout << "Window Closed" << std::endl;
	}
}
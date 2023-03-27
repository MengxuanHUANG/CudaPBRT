#pragma once

#include "Core.h"
#include "Window/Events/Event.h"

namespace CudaPBRT
{
	class Window;

	class Application
	{
	public:
		Application();
		~Application();

		void Run();
		void Terminate();

	public:
		bool OnEvent(Event& event);

		bool OnWindowClose(Event& event);

	public:
		uPtr<Window> m_MainWindow;

		bool b_Running;
		bool b_Pause;

	public:
		static Application& GetApplication()
		{
			ASSERT(s_Instance);
			return (*s_Instance);
		}

	public:
		static Application* s_Instance;
	};
}
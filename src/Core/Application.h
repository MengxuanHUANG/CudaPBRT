#pragma once

#include "Core.h"
#include "Window/Events/Event.h"
#include "LayerContainer.h"

namespace CudaPBRT
{
	class Window;
	class ImGuiLayer;

	class Application
	{
	public:
		Application();
		~Application();

		void Run();
		void Terminate();

		void PushLayer(Layer* layer);
		void PopLayer(Layer* layer, bool deleteLayer = false);

		Window* GetWindow() { return m_MainWindow.get(); }

	public:
		// dispatch event
		bool OnEvent(Event& event);

		// callback when 'x' is pressed
		bool OnWindowClose(Event& event);

	private:
		uPtr<Window> m_MainWindow;

		bool b_Running;
		bool b_Pause;

		ImGuiLayer* m_ImGuiLayer;

		LayerStack m_LayerStack;

	public:
		static Application& GetApplication()
		{
			ASSERT(s_Instance);
			return (*s_Instance);
		}

	private:
		static Application* s_Instance;
	};
}
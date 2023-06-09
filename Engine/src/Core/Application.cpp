#include "Application.h"
#include "Layer.h"
#include "Window/WindowsWindow.h"
#include "Window/Events/EventDispatcher.h"
#include "ImGui/ImGuiLayer.h"

#include <functional>

namespace CudaPBRT
{
    Application* Application::s_Instance;

	Application::Application(const WindowProps& props)
        :b_Running(true), b_Pause(false)
	{
		ASSERT(!s_Instance);

        s_Instance = this;

		// create main window
		m_MainWindow = Window::CreateWindow(std::bind(&Application::OnEvent, this, std::placeholders::_1), props);

        m_ImGuiLayer = new ImGuiLayer();
        PushLayer(m_ImGuiLayer);
	}

	Application::~Application()
	{
    }

    bool Application::OnEvent(Event& event)
    {
        EventDispatcher dispatcher(event);

        dispatcher.Dispatch<WindowCloseEvent>(std::bind(&Application::OnWindowClose, this, std::placeholders::_1));
        for (Layer* layer : m_LayerStack)
        {
            if (layer->OnEvent(event))
            {
                return true;
            }
        }
        return false;
    }

	void Application::Run()
	{
		while (b_Running)
		{
            float currentTimeStep = static_cast<float>(m_MainWindow->GetTime());

            if (!b_Pause)
            {
                for (Layer* layer : m_LayerStack)
                {
                    layer->OnUpdate(currentTimeStep);
                }
            }

            // TODO: Run PBRT on CUDA here!

            m_ImGuiLayer->Begin();
            for (Layer* layer : m_LayerStack)
            {
                layer->OnImGuiRendered(currentTimeStep);
            }
            m_ImGuiLayer->End();
            m_MainWindow->OnUpdate();
		}
	}

	void Application::Terminate()
	{
		b_Running = false;
	}

    void Application::PushLayer(Layer* layer)
    {
        ASSERT(layer);
        layer->OnAttach();

        m_LayerStack.PushLayer(layer);
    }
    
    void Application::PopLayer(Layer* layer, bool deleteLayer)
    {
        ASSERT(layer);
        layer->OnDetach();

        m_LayerStack.PopLayer(layer, deleteLayer);
    }

    bool Application::OnWindowClose(WindowCloseEvent& event)
    {
        b_Running = false;
        return false;
    }
}
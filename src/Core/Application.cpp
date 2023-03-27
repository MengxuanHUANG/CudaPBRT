#include "Application.h"
#include "Window/WindowsWindow.h"
#include "Window/Events/EventDispatcher.h"

#include <imgui/imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <functional>
#include <iostream>

ImGuiIO* io = nullptr;

void InitImGui(void* nativeWin)
{
    GLFWwindow* glfwWin = reinterpret_cast<GLFWwindow*>(nativeWin);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO();
    (void)io;
    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; //Enable Keyboard contrl
    //io->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; //Enable Gamepad contrl
    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable; //Enable Docking contrl
    io->ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; //Enable Muti-Viewport / Platform Windows
    //io->ConfigFlags |= ImGuiConfigFlags_ViewportsNoTaskBarIcons
    //io->ConfigFlags |= ImGuiConfigFlags_ViewportsNoMerge

    ImGui::StyleColorsClassic();
    ImGui_ImplGlfw_InitForOpenGL(glfwWin, true);
    ImGui_ImplOpenGL3_Init("#version 450");
}

void RenderImGui(void* nativeWin)
{
    GLFWwindow* glfwWin = reinterpret_cast<GLFWwindow*>(nativeWin);

    // Start the ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Create a window and a button
    ImGui::Begin("Hello, world!");
    if (ImGui::Button("Click me!"))
    {
        printf("Button clicked!\n");
    }
    ImGui::End();

    ImGui::Begin("new window");
    if (ImGui::Button("Click me!"))
    {
        printf("Button clicked!\n");
    }
    ImGui::End();

    // Render the ImGui frame
    ImGui::Render();

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    if (io->ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        GLFWwindow* backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }
}

namespace CudaPBRT
{
    Application* Application::s_Instance;

	Application::Application()
        :b_Running(true), b_Pause(false)
	{
		ASSERT(!s_Instance);

        s_Instance = this;

		// create main window
		m_MainWindow = Window::CreateWindow(std::bind(&Application::OnEvent, this, std::placeholders::_1));

        InitImGui(m_MainWindow->GetNativeWindow());
	}

	Application::~Application()
	{
    }

    bool Application::OnEvent(Event& event)
    {
        EventDispatcher dispatcher(event);

        dispatcher.Dispatch<WindowCloseEvent>(std::bind(&Application::OnWindowClose, this, std::placeholders::_1));
        return false;
    }

	void Application::Run()
	{
		while (b_Running)
		{
			if (!b_Pause)
			{
				m_MainWindow->OnUpdate();
                RenderImGui(m_MainWindow->GetNativeWindow());
			}
		}
	}

	void Application::Terminate()
	{
		b_Running = false;
	}

    bool Application::OnWindowClose(Event& event)
    {
        b_Running = false;
        return false;
    }
}
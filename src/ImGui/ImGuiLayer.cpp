#include "ImGuiLayer.h"

#include "Core/Application.h"
#include "Window/Window.h"

#include <imgui/imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <GLFW/glfw3.h>

namespace CudaPBRT
{
	ImGuiLayer::ImGuiLayer(const std::string& name)
		: Layer(name)
	{}

	void ImGuiLayer::OnAttach()
	{
		Application& app = Application::GetApplication();

		GLFWwindow* glfwWin = reinterpret_cast<GLFWwindow*>(app.GetWindow()->GetNativeWindow());

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
	
	void ImGuiLayer::OnDetach()
	{
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

	// should be editor only function
	void ImGuiLayer::OnImGuiRendered(float) 
	{
		static bool show = true;
		ImGui::ShowDemoWindow(&show);
	}

	void ImGuiLayer::Begin()
	{
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();

		ImGui::NewFrame();
	}

	void ImGuiLayer::End()
	{
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
}
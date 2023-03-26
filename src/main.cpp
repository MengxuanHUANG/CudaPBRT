#include <stdio.h>

#ifndef GLEW_STATIC 
#define GLEW_STATIC
#endif // !GLEW_STATIC 


#include <GL/glew.h>
#include <glfw/glfw3.h>

#include <imgui/imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "TestCuda.cuh"

GLFWwindow* window;
ImGuiIO* io = nullptr;

void errorCallback(int error, const char* description) {
	fprintf(stderr, "%s\n", description);
}

void RenderScene()
{

}

void RenderImGui()
{
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
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
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

int main()
{
	glfwSetErrorCallback(errorCallback);

	if (!glfwInit()) {
		exit(EXIT_FAILURE);
	}

	window = glfwCreateWindow(1920, 1080, "Cuda PBRT", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return 1;
	}

	glfwMakeContextCurrent(window);
	
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

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
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 450");

    while (!glfwWindowShouldClose(window))
    {
        // Poll events
        glfwPollEvents();

        RenderScene();
        RenderImGui();

        // Swap buffers
        glfwSwapBuffers(window);
    }

    return 0;
}
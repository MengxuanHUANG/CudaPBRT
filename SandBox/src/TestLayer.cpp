#include "TestLayer.h"
#include "Core/Application.h"

#include "Window/Events/EventDispatcher.h"
#include "Camera/Camera.h"
#include "PBRT/pbrt.h"

#include <GL/glew.h>
#include <imgui/imgui.h>

#include <cuda_gl_interop.h>

#include <stb_image.h>

TestLayer::TestLayer(const std::string& name)
	:Layer(name)
{
	window = Application::GetApplication().GetWindow();
	WindowProps* props = window->GetWindowProps();
	m_Camera = mkU<PerspectiveCamera>(540, 540);
	m_CamController = mkU<PerspectiveCameraController>(*m_Camera, 0.05f, 0.02f, 0.001f);
	m_CudaPBRT = mkU<CudaPathTracer>();
}

TestLayer::~TestLayer()
{
	OnDetach();
}

void TestLayer::OnAttach()
{
	WindowProps* props = window->GetWindowProps();
	int num_texels = props->width * props->height;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;

	m_CudaPBRT->InitCuda(*m_Camera);

	// load test image
	std::string filename = "E://Study//Upenn//DSGN5005 3D_Modeling//Adams.jpg";
	
	int width, height;

	unsigned char* image_data = stbi_load(filename.c_str(), &width, &height, NULL, 4);
	if (image_data)
	{
	}
	stbi_image_free(image_data);
}
void TestLayer::OnDetach()
{
	m_CudaPBRT->FreeCuda();
}

void TestLayer::OnUpdate(float delatTime)
{
	m_CudaPBRT->Run();
}

void TestLayer::OnImGuiRendered(float deltaTime)
{
	ImGuiWindowFlags window_flags = 0;
	static bool open = true;
	ImGui::Begin("Rendered Image", &open, window_flags);
	{
		if (ImGui::BeginMainMenuBar())
		{
			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("Save", "CTRL+S"))
				{
					// TODO: Show a window and save
					std::cout << "save" << std::endl;
				}
				ImGui::EndMenu();
			}
			ImGui::EndMainMenuBar();
		}
		ImGui::Image((void*)(intptr_t)(m_CudaPBRT->GetDisplayTextureId()), ImVec2(m_Camera->width, m_Camera->height));
		ImGui::End();
	}


	ImGui::Begin("Camera Control");
	if (ImGui::DragFloat3("Ref position", reinterpret_cast<float*>(&(m_Camera->ref)), 0.1f))
	{
		m_Camera->RecomputeAttributes();
		m_CudaPBRT->UpdateCamera(*m_Camera);
	}
	ImGui::End();

	bool show = true;

	ImGui::ShowDemoWindow(&show);
}

bool TestLayer::OnEvent(Event& event)
{
	EventDispatcher dispatcher(event);
	dispatcher.Dispatch<WindowResizeEvent>(std::bind(&TestLayer::OnWindowResize, this, std::placeholders::_1));
	m_CamController->OnEvent(event);

	m_CudaPBRT->UpdateCamera(*m_Camera);

	return false;
}

bool TestLayer::OnWindowResize(WindowResizeEvent& event)
{
	return false;
}
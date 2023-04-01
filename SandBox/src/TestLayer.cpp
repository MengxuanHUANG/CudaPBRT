#include "TestLayer.h"
#include "Core/Application.h"

#include "Window/Events/EventDispatcher.h"
#include "Camera/Camera.h"
#include "PBRT/pbrt.h"

#include <GL/glew.h>
#include <imgui/imgui.h>

#include <cuda_gl_interop.h>

#include <stb_image.h>

#include "PBRT/Shape/sphere.h"

TestLayer::TestLayer(const std::string& name)
	:Layer(name)
{
	window = Application::GetApplication().GetWindow();
	WindowProps* props = window->GetWindowProps();
	m_Camera = mkU<PerspectiveCamera>(400, 400);
	m_CamController = mkU<PerspectiveCameraController>(*m_Camera);
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

	// load image example
	//std::string filename = "E://Study//Upenn//DSGN5005 3D_Modeling//Adams.jpg";
	//
	//int width, height;
	//
	//unsigned char* image_data = stbi_load(filename.c_str(), &width, &height, NULL, 4);
	//if (image_data)
	//{
	//}
	//stbi_image_free(image_data);

	std::vector<ShapeData> shapes;

	// Test scene 1
	//shapes.emplace_back(ShapeType::Sphere, glm::vec3(-1, 1, 0), glm::vec3(1, 2, 1), glm::vec3(0, 0, 0));
	//shapes.emplace_back(ShapeType::Square, glm::vec3(0, 0, -5), glm::vec3(10, 10, 1), glm::vec3(0, 30, 0));

	// Test scene 2
	shapes.emplace_back(ShapeType::Sphere, glm::vec3(0, 0, 1), 0.5f * glm::vec3(2, 1, 1), glm::vec3(0, 0, 45));
	shapes.emplace_back(ShapeType::Sphere, glm::vec3(0, 0, 1), 0.5f * glm::vec3(2, 1, 1), glm::vec3(0, 0, 45));
	shapes.emplace_back(ShapeType::Square, glm::vec3(0, -0.5, 0), glm::vec3(5, 5, 1), glm::vec3(90, 0, 0));
	m_CudaPBRT->CreateShapesOnCuda(shapes);
}
void TestLayer::OnDetach()
{
	m_CudaPBRT->FreeShapesOnCuda();

	m_CudaPBRT->FreeCuda();
}

void TestLayer::OnUpdate(float delatTime)
{
	m_CudaPBRT->Run();
}

void TestLayer::OnImGuiRendered(float deltaTime)
{
	ImGui::StyleColorsLight();

	ImGui::Begin("Rendered Image");
	{
		ImGui::Image((void*)(intptr_t)(m_CudaPBRT->GetDisplayTextureId()), ImVec2(m_Camera->width, m_Camera->height));
	}
	ImGui::End();

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

	if (m_CamController->OnEvent(event))
	{
		m_CudaPBRT->UpdateCamera(*m_Camera);
	}

	return false;
}

bool TestLayer::OnWindowResize(WindowResizeEvent& event)
{
	return false;
}
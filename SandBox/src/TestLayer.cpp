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
	ShapeData data;
	data.type = ShapeType::Sphere;
	data.translation = glm::vec3(-4.f, 0, 0);
	data.rotation = glm::vec3(0.f);
	data.scale = glm::vec3(2.f);

	shapes.push_back(data);
	data.scale = glm::vec3(1.f);
	data.translation = glm::vec3(2.f, 0, 0);
	shapes.push_back(data);
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
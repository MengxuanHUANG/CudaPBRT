#include "TestLayer.h"
#include "Core/Application.h"

#include "Window/Events/EventDispatcher.h"
#include "Camera/Camera.h"
#include "PBRT/pbrt.h"
#include "PBRT/scene.h"

#include <GL/glew.h>
#include <imgui/imgui.h>

#include <cuda_gl_interop.h>

#include <stb_image.h>
#include <stb_image_write.h>

#include "PBRT/Shape/sphere.h"
#include "PBRT/Material/material.h"
#include "PBRT/Light/light.h"

TestLayer::TestLayer(const std::string& name)
	:Layer(name)
{
	window = Application::GetApplication().GetWindow();
	WindowProps* props = window->GetWindowProps();
	m_Camera = mkU<PerspectiveCamera>(680, 680, 19.5f, glm::vec3(0, 5.5, -30), glm::vec3(0, 2.5, 0));
	m_CamController = mkU<PerspectiveCameraController>(*m_Camera);
	m_Scene = mkU<Scene>();
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

	m_CudaPBRT = mkU<CudaPathTracer>();
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

	// Hard-coding Cornell Box Scene
	// shape data
	std::vector<ShapeData> shapeData;
	int matteWhiteId	= 0;
	int matteRedId		= 1;
	int matteGreenId	= 2;
	int mirrorId		= 3;
	int glassId			= 4;
	shapeData.emplace_back(ShapeType::Square, matteWhiteId, glm::vec3(0, -2.5, 0), glm::vec3(10, 10, 1), glm::vec3(-90, 0, 0)); // Floor
	shapeData.emplace_back(ShapeType::Square, matteRedId,   glm::vec3(5, 2.5, 0),  glm::vec3(10, 10, 1), glm::vec3(0, -90, 0)); // Red wall
	shapeData.emplace_back(ShapeType::Square, matteGreenId, glm::vec3(-5, 2.5, 0), glm::vec3(10, 10, 1), glm::vec3(0, 90, 0)); // Green Wall
	shapeData.emplace_back(ShapeType::Square, matteWhiteId, glm::vec3(0, 2.5, 5),  glm::vec3(10, 10, 1), glm::vec3(0, 180, 0)); // Back Wall
	shapeData.emplace_back(ShapeType::Square, matteWhiteId, glm::vec3(0, 7.5, 0),  glm::vec3(10, 10, 1), glm::vec3(90, 0, 0)); // Ceiling

	shapeData.emplace_back(ShapeType::Sphere, glassId, glm::vec3(0, 1.25, 0), glm::vec3(3, 3, 3), glm::vec3(0, 0, 0));

	//shapeData.emplace_back(ShapeType::Cube, matteWhiteId, glm::vec3(2, 0, 3), glm::vec3(3, 6, 3), glm::vec3(0, 27.5, 0)); // Long Cube
	//shapeData.emplace_back(ShapeType::Cube, matteWhiteId, glm::vec3(-2, -1, 0.75), glm::vec3(3, 3, 3), glm::vec3(0, -17.5, 0)); // Short Cube
	
	// material data
	std::vector<MaterialData> materialData;
	materialData.emplace_back(MaterialType::DiffuseReflection, glm::vec3(0.85, 0.81, 0.78)); //matteWhite
	materialData.emplace_back(MaterialType::DiffuseReflection, glm::vec3(0.63, 0.065, 0.05)); //matteRed
	materialData.emplace_back(MaterialType::DiffuseReflection, glm::vec3(0.14, 0.45, 0.091)); //matteGreen
	materialData.emplace_back(MaterialType::SpecularReflection, glm::vec3(1.f, 1.f, 1.f)); // mirror
	materialData.emplace_back(MaterialType::SpecularTransmission, glm::vec3(0.9f, 0.9f, 1.f), 0.f, 0.f, 1.55f); // glass

	// Light
	std::vector<LightData> lightData;
	ShapeData areaLightShape(ShapeType::Square, -1, glm::vec3(0, 7.45, 0), glm::vec3(3, 3, 1), glm::vec3(90, 0, 0));
	Spectrum Le(40);
	lightData.emplace_back(LightType::ShapeLight, areaLightShape, Le);

	CreateArrayOnCude<Shape, ShapeData>(m_Scene->shapes, m_Scene->shape_count, shapeData);
	CreateArrayOnCude<Material, MaterialData>(m_Scene->materials, m_Scene->material_count, materialData);
	CreateArrayOnCude<Light, LightData>(m_Scene->lights, m_Scene->light_count, lightData);

	
}
void TestLayer::OnDetach()
{
	m_Scene->FreeDataOnCuda();
	m_CudaPBRT.release();
}

void TestLayer::OnUpdate(float delatTime)
{
	m_CudaPBRT->Run(m_Scene.get());
}

void TestLayer::OnImGuiRendered(float deltaTime)
{
	ImGui::StyleColorsLight();
	bool open = true;
	ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoMove;
	ImGui::Begin("Rendered Image", &open, window_flags);
	{
		ImGui::Text("Iteration: %d", m_CudaPBRT->m_Iteration);
		ImGui::Image((void*)(intptr_t)(m_CudaPBRT->GetDisplayTextureId()), ImVec2(m_Camera->width, m_Camera->height));
	}
	ImGui::End();

	ImGui::Begin("Camera Control");
	if (ImGui::DragFloat3("Ref position", reinterpret_cast<float*>(&(m_Camera->ref)), 0.1f))
	{
		m_Camera->RecomputeAttributes();
		m_CudaPBRT->UpdateCamera(*m_Camera);
		m_CudaPBRT->ResetPRBT();
	}
	if (ImGui::Button("Save Image"))
	{
		stbi_write_png("C://Users//admas//Downloads//save.png", m_Camera->width, m_Camera->height, 4, m_CudaPBRT->host_image, m_Camera->width * 4);
	}
	ImGui::End();

	//bool show = true;
	//
	//ImGui::ShowDemoWindow(&show);
}

bool TestLayer::OnEvent(Event& event)
{
	EventDispatcher dispatcher(event);
	dispatcher.Dispatch<WindowResizeEvent>(std::bind(&TestLayer::OnWindowResize, this, std::placeholders::_1));

	if (m_CamController->OnEvent(event))
	{
		m_CudaPBRT->UpdateCamera(*m_Camera);
		m_CudaPBRT->ResetPRBT();
	}

	return false;
}

bool TestLayer::OnWindowResize(WindowResizeEvent& event)
{
	return false;
}
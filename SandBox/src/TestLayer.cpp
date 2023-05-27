#include "TestLayer.h"
#include "Core/Application.h"

#include "Window/Window.h"
#include "Window/Events/EventDispatcher.h"

#include "Camera/Camera.h"
#include "Camera/CameraController.h"

#include "PBRT/pbrt.h"
#include "PBRT/BVH/boundingBox.h"
#include "PBRT/Shape/triangle.h"
#include "PBRT/BVH/bvh.h"
#include "PBRT/texture.h"

#include "PBRT/cpuScene.h"


#include <GL/glew.h>
#include <imgui/imgui.h>

#include <cuda_gl_interop.h>

#include <stb_image.h>
#include <stb_image_write.h>

#include "PBRT/Shape/sphere.h"
#include "PBRT/Material/material.h"
#include "PBRT/Light/light.h"
#include "MeshLoader/meshLoader.h"

#include <iomanip>
#include <format>
#include <iostream>
#include <fstream>
#include <ranges>
#include <string>
#include <string_view>

TestLayer::TestLayer(const std::string& name)
	:Layer(name)
{
	window = Application::GetApplication().GetWindow();
	WindowProps* props = window->GetWindowProps();
	
	m_Scene = mkU<CPUScene>();
}

TestLayer::~TestLayer()
{
	OnDetach();
}

void TestLayer::OnAttach()
{
	m_Scene->LoadSceneFromJSON("E://Projects//CUDA_Projects//CudaPBRT//res//scenes//Cornel.json");

	m_CudaPBRT = mkU<CudaPathTracer>();
	m_CudaPBRT->InitCuda(*(m_Scene->camera));

	m_SelectedMaterial = m_Scene->materialData.size() - 1;

	std::ifstream in("E://Projects//CUDA_Projects//CudaPBRT//res//scenes//Cornel.json");
	if (in.is_open())
	{
		JSON json = JSON::parse(in);

		JSON frames_arr = json["frames"];
		JSON scene_arr = frames_arr[0]["scene"];
		//std::cout << scene_arr["camera"]["width"] << std::endl;
		std::cout << frames_arr[0]["frameNumber"] << std::endl;
	}
}
void TestLayer::OnDetach()
{
	m_Scene->ClearScene();
	m_CudaPBRT.release();
}

void TestLayer::OnUpdate(float delatTime)
{
	float time_step = window->GetTime();
	m_CudaPBRT->Run(&(m_Scene->m_GPUScene));
	m_FrameTime = window->GetTime() - time_step;
}

void TestLayer::OnImGuiRendered(float deltaTime)
{
	PerspectiveCamera& camera = *(m_Scene->camera);

	ImGui::StyleColorsLight();
	bool open = true;
	ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoMove;
	ImGui::Begin("Rendered Image", &open, window_flags);
	{
		ImGui::Text("Iteration: %d", m_CudaPBRT->m_Iteration);
		ImGui::Text("Frame Time: %f", m_FrameTime);
		ImGui::Text("fps: %f", 1.f / m_FrameTime);
		ImGui::Image((void*)(intptr_t)(m_CudaPBRT->GetDisplayTextureId()), ImVec2(camera.width, camera.height));
	}
	ImGui::End();

	ImGui::Begin("Camera Control");
	if (ImGui::DragFloat3("Ref position", reinterpret_cast<float*>(&(camera.ref)), 0.1f))
	{
		camera.RecomputeAttributes();
		m_CudaPBRT->UpdateCamera(camera);
		m_CudaPBRT->ResetPRBT();
	}
	if (ImGui::Button("Save Image"))
	{
		stbi_write_png("C://Users//admas//Downloads//save.png", camera.width, camera.height, 4, m_CudaPBRT->host_image, camera.width * 4);
	}
	ImGui::End();

	ImGui::Begin("Material Editor");
	MaterialData& mdata = m_Scene->materialData[m_SelectedMaterial];

	bool is_edited = false;
	is_edited |= ImGui::ColorEdit3("Albedo", reinterpret_cast<float*>(&(mdata.albedo)));
	is_edited |= ImGui::DragFloat("Metallic", &(mdata.metallic), 0.01f, 0.f, 1.f);
	is_edited |= ImGui::DragFloat("Roughness", &(mdata.roughness), 0.01f, 0.f, 1.f);

	if (is_edited)
	{
		UpdateArrayOnCuda<Material, MaterialData>(m_Scene->m_GPUScene.materials, m_Scene->materialData, m_SelectedMaterial, m_SelectedMaterial + 1);
		m_CudaPBRT->ResetPRBT();
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

	if (m_Scene->camController->OnEvent(event))
	{
		m_CudaPBRT->UpdateCamera(*(m_Scene->camera));
		m_CudaPBRT->ResetPRBT();
	}

	return false;
}

bool TestLayer::OnWindowResize(WindowResizeEvent& event)
{
	return false;
}

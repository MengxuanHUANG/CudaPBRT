#include "TestLayer.h"
#include "Core/Application.h"

#include "Window/Events/EventDispatcher.h"
#include "Camera/Camera.h"
#include "PBRT/pbrt.h"
#include "PBRT/BVH/boundingBox.h"
#include "PBRT/Shape/triangle.h"
#include "PBRT/BVH/bvh.h"
#include "PBRT/texture.h"

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
#include <string>
#include <iostream>
#include <fstream>

TestLayer::TestLayer(const std::string& name)
	:Layer(name)
{
	window = Application::GetApplication().GetWindow();
	WindowProps* props = window->GetWindowProps();
	m_Camera = mkU<PerspectiveCamera>(680, 680, 19.5f, glm::vec3(0, 5.5, -30), glm::vec3(0, 2.5, 0));
	m_CamController = mkU<PerspectiveCameraController>(*m_Camera);
	m_Scene = mkU<CPUScene>();
}

TestLayer::~TestLayer()
{
	OnDetach();
}

void TestLayer::OnAttach()
{
	WindowProps* props = window->GetWindowProps();

	m_CudaPBRT = mkU<CudaPathTracer>();
	m_CudaPBRT->InitCuda(*m_Camera);

	m_Scene->LoadSceneFromJSON("");
	
	m_SelectedMaterial = m_Scene->materialData.size() - 1;
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
	ImGui::StyleColorsLight();
	bool open = true;
	ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoMove;
	ImGui::Begin("Rendered Image", &open, window_flags);
	{
		ImGui::Text("Iteration: %d", m_CudaPBRT->m_Iteration);
		ImGui::Text("Frame Time: %f", m_FrameTime);
		ImGui::Text("fps: %f", 1.f / m_FrameTime);
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

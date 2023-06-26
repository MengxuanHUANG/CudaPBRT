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

std::string TestLayer::JSON_PATH = "E://Projects//CUDA_Projects//CudaPBRT//res//scenes//";
std::string TestLayer::IMG_SAVE_PATH = "C://Users//admas//Downloads//";

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
	m_CurrentFile = "Castle.json";

	m_Scene->LoadSceneFromJsonFile((JSON_PATH + m_CurrentFile).c_str());
	m_Scene->m_GPUScene.M = 1;

	m_CudaPBRT = mkU<CudaPathTracer>();
	m_CudaPBRT->InitCuda(*(m_Scene->camera));

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
	PerspectiveCamera& camera = *(m_Scene->camera);

	ImGui::StyleColorsLight();
	bool open = true;
	static bool auto_save = false;
	static int auto_save_it = 1;

	ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoMove;
	ImGui::Begin("Rendered Image", &open, window_flags);
	{
		ImGui::Text("Iteration: %d", m_CudaPBRT->m_Iteration);
		ImGui::Text("Frame Time: %f", m_FrameTime);
		ImGui::Text("fps: %f", 1.f / m_FrameTime);
		
		bool is_edited = false;
		is_edited |= ImGui::Button("Reset PT");
		is_edited |= ImGui::DragInt("M", &(m_Scene->m_GPUScene.M), 1, 1, 20);
		is_edited |= ImGui::DragInt("SpatialReuse Count", &(m_Scene->m_GPUScene.spatialReuseCount), 1, 1, 20);
		is_edited |= ImGui::DragFloat("SpatialReuse Radius", &(m_Scene->m_GPUScene.spatialReuseRadius), 1.f, 0.f, 30.f);
		is_edited |= ImGui::Checkbox("Temporal Reuse", &(m_Scene->m_GPUScene.temporalReuse));
		is_edited |= ImGui::Checkbox("Spatial Reuse", &(m_Scene->m_GPUScene.spatialReuse));
		ImGui::Image((void*)(intptr_t)(m_CudaPBRT->GetDisplayTextureId()), ImVec2(camera.width, camera.height));
		if (is_edited) m_CudaPBRT->ResetPRBT();
	}
	ImGui::End();
	
	ImGui::Begin("Save Image", &open, window_flags);
	{
		ImGui::Checkbox("Auto Save", &auto_save);
		ImGui::InputInt("AUto Save Iteration", &auto_save_it);
		if (ImGui::Button("Save Image") || (auto_save && m_CudaPBRT->m_Iteration == auto_save_it))
		{
			std::string str = std::vformat("_M{}_it{}", std::make_format_args(m_Scene->m_GPUScene.M, m_CudaPBRT->m_Iteration));
			std::string extension = ".png";

			stbi_write_png((IMG_SAVE_PATH + m_CurrentFile + str + extension).c_str(), camera.width, camera.height, 4, m_CudaPBRT->host_image, camera.width * 4);

			float3* host_hdr = new float3[camera.width * camera.height];

			// Copy rendered result to CPU.
			//cudaMemcpy(host_hdr, m_CudaPBRT->device_hdr_image, sizeof(float3) * camera.width * camera.height, cudaMemcpyDeviceToHost);
			//CUDA_CHECK_ERROR();
			//
			//stbi_write_hdr("C://Users//admas//Downloads//save.hdr", camera.width, camera.height, 3, reinterpret_cast<float*>(&(host_hdr[0].x)));
			//
			//if (host_hdr)
			//{
			//	delete[] host_hdr;
			//}
		}
	}
	ImGui::End();

	ImGui::Begin("Camera Control");
	{
		bool cam_is_edited = false;
		cam_is_edited |= ImGui::DragFloat3("Ref position", reinterpret_cast<float*>(&(camera.ref)), 0.1f);
		cam_is_edited |= ImGui::DragFloat("Len Radius", &(camera.lensRadius), 0.01f, 0.f, 0.5f);
		cam_is_edited |= ImGui::DragFloat("Focal Distance", &(camera.focalDistance), 0.1f, 1.f, 100.f);

		if (cam_is_edited)
		{
			camera.RecomputeAttributes();
			m_CudaPBRT->UpdateCamera(camera);
			m_CudaPBRT->ResetPRBT();
		}
	}
	ImGui::End();

	ImGui::Begin("Material Editor");
	{
		MaterialData& mdata = m_Scene->materialData[m_SelectedMaterial];

		bool is_edited = false;
		is_edited |= ImGui::ColorEdit3("Albedo", reinterpret_cast<float*>(&(mdata.albedo)));
		is_edited |= ImGui::DragFloat("Metallic", &(mdata.metallic), 0.01f, 0.f, 1.f);
		is_edited |= ImGui::DragFloat("Roughness", &(mdata.roughness), 0.01f, 0.f, 1.f);
		is_edited |= ImGui::DragFloat("Lv", &(mdata.Lv), 0.1f, 0.f, 100.f);

		if (is_edited)
		{
			UpdateArrayOnCuda<Material, MaterialData>(m_Scene->m_GPUScene.materials, m_Scene->materialData, m_SelectedMaterial, m_SelectedMaterial + 1);
			m_CudaPBRT->ResetPRBT();
		}
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

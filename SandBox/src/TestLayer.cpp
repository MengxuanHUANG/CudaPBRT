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

#include <iomanip>
#include <format>
#include <string>
#include <iostream>
#include <fstream>
#include <ranges>
#include <string_view>

TestLayer::TestLayer(const std::string& name)
	:Layer(name)
{
	window = Application::GetApplication().GetWindow();
	WindowProps* props = window->GetWindowProps();
	m_Camera = mkU<PerspectiveCamera>(680, 680, 19.5f, glm::vec3(0, 5.5, -30), glm::vec3(0, 2.5, 0));
	m_CamController = mkU<PerspectiveCameraController>(*m_Camera);
	m_Scene = mkU<GPUScene>();
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

	LoadScene();
	
	m_SelectedMaterial = materialData.size() - 1;

	//m_CudaPBRT->DisplayTexture(*m_Textures[2]);
}
void TestLayer::OnDetach()
{
	shapeData.clear();
	materialData.clear();
	lightData.clear();

	m_Scene->FreeDataOnCuda();
	m_CudaPBRT.release();
}

void TestLayer::OnUpdate(float delatTime)
{
	float time_step = window->GetTime();
	m_CudaPBRT->Run(m_Scene.get());
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
	MaterialData& mdata = materialData[m_SelectedMaterial];

	bool is_edited = false;
	is_edited |= ImGui::ColorEdit3("Albedo", reinterpret_cast<float*>(&(mdata.albedo)));
	is_edited |= ImGui::DragFloat("Metallic", &(mdata.metallic), 0.01f, 0.f, 1.f);
	is_edited |= ImGui::DragFloat("Roughness", &(mdata.roughness), 0.01f, 0.f, 1.f);

	if (is_edited)
	{
		UpdateArrayOnCuda<Material, MaterialData>(m_Scene->materials, materialData, m_SelectedMaterial, m_SelectedMaterial + 1);
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

void TestLayer::TestSingleTriangle(std::vector<ShapeData>& shapeData,
								   std::vector<TriangleData>& triangles,
								   std::vector<glm::vec3>& vertices,
								   std::vector<glm::vec3>& normals,
								   std::vector<glm::vec2>& uvs)
{
	size_t v_start_id = vertices.size();

	vertices.emplace_back( 1,  1,  1);
	vertices.emplace_back( 1, -1,  1);
	vertices.emplace_back(-1, -1,  1);
	vertices.emplace_back(-1,  1,  1);

	uvs.emplace_back(0, 0);
	uvs.emplace_back(0, 1);
	uvs.emplace_back(1, 0);
	uvs.emplace_back(1, 1);

	triangles.emplace_back(glm::ivec3(0, 1, 2), glm::ivec3(-1), glm::ivec3(0, 1, 3));
	triangles.emplace_back(glm::ivec3(0, 2, 3), glm::ivec3(-1), glm::ivec3(0, 3, 2));
	glm::mat4 transform;
	Shape::ComputeTransform(glm::vec3(0, 0, 0), glm::vec3(0, 0, 0), glm::vec3(3, 1, 1), transform);

	for (size_t i = v_start_id; i < vertices.size(); ++i)
	{
		vertices[i] = glm::vec3(transform * glm::vec4(vertices[i], 1.f));
	}
}

void TestLayer::AddCornellBox_Triangles(std::vector<ShapeData>& shapeData,
										std::vector<TriangleData>& triangles,
										std::vector<glm::vec3>& vertices,
										std::vector<glm::vec3>& normals,
										std::vector<glm::vec2>& uvs, int material_a, int material_b)
{
	std::vector<int> v_start_id;
	std::vector<int> tri_end_id;
	std::vector<int> material_id;

	// apply transform
	glm::mat4 transform;

	v_start_id.emplace_back(vertices.size());
	vertices.emplace_back(1, 1, -1);
	vertices.emplace_back(1, -1, -1);
	vertices.emplace_back(-1, -1, -1);
	vertices.emplace_back(-1, 1, -1);
	
	vertices.emplace_back(1, 1, 1);
	vertices.emplace_back(1, -1, 1);
	vertices.emplace_back(-1, -1, 1);
	vertices.emplace_back(-1, 1, 1);
	
	triangles.emplace_back(glm::ivec3(0, 1, 2)); // front
	triangles.emplace_back(glm::ivec3(0, 2, 3)); // front
	triangles.emplace_back(glm::ivec3(5, 4, 7)); // back
	triangles.emplace_back(glm::ivec3(5, 7, 6)); // back
	triangles.emplace_back(glm::ivec3(6, 7, 3)); // right
	triangles.emplace_back(glm::ivec3(6, 3, 2)); // right
	triangles.emplace_back(glm::ivec3(0, 5, 1)); // left
	triangles.emplace_back(glm::ivec3(0, 4, 5)); // left
	triangles.emplace_back(glm::ivec3(3, 7, 4)); // top
	triangles.emplace_back(glm::ivec3(3, 4, 0)); // top
	triangles.emplace_back(glm::ivec3(2, 1, 5)); // bottom
	triangles.emplace_back(glm::ivec3(2, 5, 6)); // bottom
	tri_end_id.emplace_back(triangles.size());
	material_id.emplace_back(material_a);
	
	Shape::ComputeTransform(glm::vec3(2, 0, 3), glm::vec3(0, 27.5, 0), glm::vec3(1.5, 3, 1.5), transform);

	for (int i = v_start_id.back(); i < vertices.size(); ++i)
	{
		vertices[i] = glm::vec3(transform * glm::vec4(vertices[i], 1.f));
	}

	v_start_id.emplace_back(vertices.size());
	vertices.emplace_back(1, 1, -1);
	vertices.emplace_back(1, -1, -1);
	vertices.emplace_back(-1, -1, -1);
	vertices.emplace_back(-1, 1, -1);

	vertices.emplace_back(1, 1, 1);
	vertices.emplace_back(1, -1, 1);
	vertices.emplace_back(-1, -1, 1);
	vertices.emplace_back(-1, 1, 1);

	triangles.emplace_back(8 + glm::ivec3(0, 1, 2)); // front
	triangles.emplace_back(8 + glm::ivec3(0, 2, 3)); // front
	triangles.emplace_back(8 + glm::ivec3(5, 4, 7)); // back
	triangles.emplace_back(8 + glm::ivec3(5, 7, 6)); // back
	triangles.emplace_back(8 + glm::ivec3(6, 7, 3)); // right
	triangles.emplace_back(8 + glm::ivec3(6, 3, 2)); // right
	triangles.emplace_back(8 + glm::ivec3(0, 5, 1)); // left
	triangles.emplace_back(8 + glm::ivec3(0, 4, 5)); // left
	triangles.emplace_back(8 + glm::ivec3(3, 7, 4)); // top
	triangles.emplace_back(8 + glm::ivec3(3, 4, 0)); // top
	triangles.emplace_back(8 + glm::ivec3(2, 1, 5)); // bottom
	triangles.emplace_back(8 + glm::ivec3(2, 5, 6)); // bottom
	tri_end_id.emplace_back(triangles.size());
	material_id.emplace_back(material_b);

	Shape::ComputeTransform(glm::vec3(-2, -1, 0.75), glm::vec3(0, -17.5, 0), glm::vec3(1.5, 1.5, 1.5), transform);

	for (int i = v_start_id.back(); i < vertices.size(); ++i)
	{
		vertices[i] = glm::vec3(transform * glm::vec4(vertices[i], 1.f));
	}
}

void TestLayer::CreateBoundingBox(std::vector<ShapeData>& shapeData, std::vector<glm::vec3>& vertices)
{
	std::vector<BoundingBox> boundingBoxes;
	std::vector<BVHNode> bvh_nodes;

	CreateBVH(shapeData, vertices, boundingBoxes, bvh_nodes);

	BufferData<BoundingBox>(m_Scene->boundings, boundingBoxes.data(), boundingBoxes.size());
	BufferData<BVHNode>(m_Scene->BVH, bvh_nodes.data(), bvh_nodes.size());
}

void TestLayer::LoadScene()
{
	m_Scene->FreeDataOnCuda();

	// load texture
	const char* env_map_path = "E://Projects//CUDA_Projects//CudaPBRT//res//environment_maps//interior_atelier_soft_daylight.hdr";

	const char* tex_albedo_path = "E://Projects//CUDA_Projects//CudaPBRT//res//textures//whitebubble.jpg";
	const char* tex_nor_path = "E://Projects//CUDA_Projects//CudaPBRT//res//textures//whitebubbleN.jpg";

	const char* cam_albedo_path = "E://Projects//CUDA_Projects//CudaPBRT//res//textures//#CAM0001_Textures_COL_2k.png";
	const char* cam_nor_path = "E://Projects//CUDA_Projects//CudaPBRT//res//textures//#CAM0001_Textures_NRML_2k.png";
	const char* cam_metl_path = "E://Projects//CUDA_Projects//CudaPBRT//res//textures//#CAM0001_Textures_METL_2k.png";
	const char* cam_rougn_path = "E://Projects//CUDA_Projects//CudaPBRT//res//textures//#CAM0001_Textures_ROUGH_2k.png";
	m_Textures.emplace_back(CudaTexture::CreateCudaTexture(cam_albedo_path, true));
	m_Textures.emplace_back(CudaTexture::CreateCudaTexture(cam_nor_path, true));
	m_Textures.emplace_back(CudaTexture::CreateCudaTexture(cam_metl_path, true));
	m_Textures.emplace_back(CudaTexture::CreateCudaTexture(cam_rougn_path, true));
	
	m_Textures.emplace_back(CudaTexture::CreateCudaTexture(env_map_path, true));

	m_Scene->envMap.SetTexObj(m_Textures.back()->GetTextureObject());

	// Hard-coding Cornell Box Scene

	// material data
	int matteWhiteId	= 0;
	int matteRedId		= 1;
	int matteGreenId	= 2;
	int mirrorId		= 3;
	int glassId			= 4;
	int obj_tex_Id		= 5;
	int microfacetId	= 6;
	int MetallicWfId	= 7;

	materialData.emplace_back(MaterialType::LambertianReflection, glm::vec3(0.85, 0.81, 0.78)); //matteWhite
	materialData.emplace_back(MaterialType::LambertianReflection, glm::vec3(0.63, 0.065, 0.05)); //matteRed
	materialData.emplace_back(MaterialType::LambertianReflection, glm::vec3(0.14, 0.45, 0.091)); //matteGreen
	materialData.emplace_back(MaterialType::SpecularReflection, glm::vec3(1.f, 1.f, 1.f)); // mirror
	materialData.emplace_back(MaterialType::SpecularTransmission, glm::vec3(.9f, .9f, 1.f), 0.f, 0.f, 1.55f); // glass
	materialData.emplace_back(MaterialType::MetallicWorkflow, m_Textures[0]->GetTextureObject(),
															  m_Textures[1]->GetTextureObject(),
															  m_Textures[2]->GetTextureObject(),
															  m_Textures[3]->GetTextureObject()); // texture MetallicWorkflow
	materialData.emplace_back(MaterialType::MicrofacetReflection, glm::vec3(.8f, .8f, .8f), .5f, 0.5f); // microfacet
	materialData.emplace_back(MaterialType::MetallicWorkflow, glm::vec3(.8f, .8f, .8f), 1.0f, 0.0f); // MetallicWorkflow

	//shapeData.emplace_back(ShapeType::Square, matteWhiteId, glm::vec3(0, -2.5, 0), glm::vec3(-90, 0, 0), glm::vec3(10, 10, 1)); // Floor
	//shapeData.emplace_back(ShapeType::Square, matteRedId, glm::vec3(5, 2.5, 0), glm::vec3(0, -90, 0), glm::vec3(10, 10, 1)); // Red wall
	//shapeData.emplace_back(ShapeType::Square, matteGreenId, glm::vec3(-5, 2.5, 0), glm::vec3(0, 90, 0), glm::vec3(10, 10, 1)); // Green Wall
	//shapeData.emplace_back(ShapeType::Square, microfacetId, glm::vec3(0, 2.5, 5), glm::vec3(0, 180, 0), glm::vec3(10, 10, 1)); // Back Wall
	//shapeData.emplace_back(ShapeType::Square, matteWhiteId, glm::vec3(0, 7.5, 0), glm::vec3(90, 0, 0), glm::vec3(10, 10, 1)); // Ceiling

	std::vector<glm::vec3> vertices;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec2> uvs;
	std::vector<TriangleData> triangles;

	//TestSingleTriangle(shapeData, triangles, vertices, normals, uvs);
	//AddCornellBox_Triangles(shapeData, triangles, vertices, normals, uvs, matteWhiteId, matteWhiteId);

	//shapeData.emplace_back(ShapeType::Sphere, MetallicWfId, glm::vec3(0, 1.25, 0), glm::vec3(0, 0, 0), glm::vec3(2, 2, 2));
	//shapeData.emplace_back(ShapeType::Cube, glassId, glm::vec3(2, 0, 3), glm::vec3(0, 27.5, 0), glm::vec3(3, 6, 3)); // Long Cube
	//shapeData.emplace_back(ShapeType::Cube, matteWhiteId, glm::vec3(-2, -1, 0.75), glm::vec3(0, -17.5, 0), glm::vec3(3, 3, 3)); // Short Cube
	LoadObj(shapeData, triangles, vertices, normals, uvs, "E://Projects//CUDA_Projects//CudaPBRT//res//models//Camera.obj");

	BufferData<glm::vec3>(m_Scene->vertices, vertices.data(), vertices.size());
	BufferData<glm::vec3>(m_Scene->normals, normals.data(), normals.size());
	BufferData<glm::vec2>(m_Scene->uvs, uvs.data(), uvs.size());

	for (int i = 0; i < triangles.size(); ++i)
	{
		shapeData.emplace_back(obj_tex_Id, triangles[i], m_Scene->vertices, m_Scene->normals, m_Scene->uvs);
	}

	CreateBoundingBox(shapeData, vertices);
	
	// Lights
	ShapeData areaLightShape(ShapeType::Square, -1, glm::vec3(0, 7.45, 0), glm::vec3(90, 0, 0), glm::vec3(3, 3, 1));
	Spectrum Le(40);
	//lightData.emplace_back(LightType::ShapeLight, areaLightShape, Le);

	CreateArrayOnCude<Shape, ShapeData>(m_Scene->shapes, m_Scene->shape_count, shapeData);
	CreateArrayOnCude<Material, MaterialData>(m_Scene->materials, m_Scene->material_count, materialData);
	CreateArrayOnCude<Light, LightData>(m_Scene->lights, m_Scene->light_count, lightData);
}

void TestLayer::LoadObj(std::vector<ShapeData>& shapeData,
						std::vector<TriangleData>& triangles,
						std::vector<glm::vec3>& vertices, 
						std::vector<glm::vec3>& normals, 
						std::vector<glm::vec2>& uvs,
						const char* path)
{
	std::ifstream in(path, std::ios::in);

	size_t v_start_id = vertices.size();
	size_t n_start_id = normals.size();

	int line_count = 0;
	int f_start = 1;
	for (std::string line; std::getline(in, line);)
	{
		//std::cout << line << std::endl;
		if (line.starts_with("FStart"))
		{
			std::string delim = " ";

			auto v = line | std::views::split(delim) | std::views::transform([](auto word) {
				return std::string(word.begin(), word.end());
			});

			std::vector<std::string> result(v.begin(), v.end());

			f_start = std::stoi(result[1]);
		}
		if (line.starts_with("vn"))
		{
			std::string delim = " ";

			auto v = line | std::views::split(delim) | std::views::transform([](auto word) {
				return std::string(word.begin(), word.end());
				});

			std::vector<std::string> result(v.begin(), v.end());

			normals.emplace_back(std::stof(result[1]), std::stof(result[2]), std::stof(result[3]));
		}
		else if (line.starts_with("vt"))
		{
			std::string delim = " ";

			auto v = line | std::views::split(delim) | std::views::transform([](auto word) {
				return std::string(word.begin(), word.end());
				});

			std::vector<std::string> result(v.begin(), v.end());

			uvs.emplace_back(std::stof(result[1]), std::stof(result[2]));
		}
		else if (line.starts_with("v"))
		{
			std::string delim = " ";

			auto v = line | std::views::split(delim) | std::views::transform([](auto word) {
				return std::string(word.begin(), word.end());
			});

			std::vector<std::string> result(v.begin(), v.end());

			vertices.emplace_back(std::stof(result[1]), std::stof(result[2]), std::stof(result[3]));
		}
		else if (line.starts_with("f"))
		{
			std::string delim = " ";

			auto v = line | std::views::split(delim) | std::views::transform([](auto word) {
				return std::string(word.begin(), word.end());
			});

			std::vector<std::string> result(v.begin(), v.end());
			
			ASSERT(result.size() > 3);
			
			std::vector<int> v_id;
			std::vector<int> n_id;
			std::vector<int> uv_id;
			
			v_id.resize(result.size() - 1);
			n_id.resize(result.size() - 1);
			uv_id.resize(result.size() - 1);

			for (int i = 0; i < result.size() - 1; ++i)
			{
				auto id_str = result[i + 1] | std::views::split('/') | std::views::transform([](auto word) {
					return std::string(word.begin(), word.end());
					});
				std::vector<std::string> ids(id_str.begin(), id_str.end());
				v_id[i] = std::stoi(ids[0]) - f_start;
				uv_id[i] = std::stoi(ids[1]) - f_start;
				n_id[i] = std::stoi(ids[2]) - f_start;
			}
			// naive triangulation
			for (int i = 1; i < result.size() - 2; ++i)
			{
				triangles.emplace_back(glm::ivec3( v_id[0],  v_id[i],  v_id[i + 1]),
									   glm::ivec3( n_id[0],  n_id[i],  n_id[i + 1]),
									   glm::ivec3(uv_id[0], uv_id[i], uv_id[i + 1]));
			}
		}
	}

	glm::mat4 transform;
	glm::mat3 transposeInvT;
	Shape::ComputeTransform(glm::vec3(0, 0, 0), glm::vec3(0, 180, 0), glm::vec3(0.5), transform);
	transposeInvT = glm::transpose(glm::inverse(glm::mat3(transform)));

	for (size_t i = v_start_id; i < vertices.size(); ++i)
	{
		vertices[i] = glm::vec3(transform * glm::vec4(vertices[i], 1.f));
	}

	for (size_t i = n_start_id; i < normals.size(); ++i)
	{
		normals[i] = glm::normalize(transposeInvT * normals[i]);
	}
}
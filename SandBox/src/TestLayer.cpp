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
	m_Camera = mkU<PerspectiveCamera>(1200, 600, 19.5f, glm::vec3(0, 5.5, -30), glm::vec3(0, 2.5, 0));
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

	LoadScene();

	m_CudaPBRT->DisplayTexture(*m_Textures[2]);
}
void TestLayer::OnDetach()
{
	m_Scene->FreeDataOnCuda();
	m_CudaPBRT.release();
}

void TestLayer::OnUpdate(float delatTime)
{
	//float time_step = window->GetTime();
	//m_CudaPBRT->Run(m_Scene.get());
	//m_FrameTime = window->GetTime() - time_step;
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

void TestLayer::TestSingleTriangle(std::vector<ShapeData>& shapeData, std::vector<glm::vec3>& vertices)
{
	std::vector<glm::ivec3> triangles;

	vertices.emplace_back( 0,  1, -2);
	vertices.emplace_back( 1, -1, -2);
	vertices.emplace_back(-1, -1, -2);
	vertices.emplace_back( 0,  1, 2);
	vertices.emplace_back( 1, -1, 2);
	vertices.emplace_back(-1, -1, 2);
	triangles.emplace_back(glm::ivec3(0, 1, 2));
	triangles.emplace_back(glm::ivec3(5, 4, 3));

	BufferData<glm::vec3>(m_Scene->vertices, vertices.data(), vertices.size());
	shapeData.emplace_back(4, triangles[0], m_Scene->vertices);
	shapeData.emplace_back(4, triangles[1], m_Scene->vertices);
}

void TestLayer::AddCornellBox_Triangles(std::vector<ShapeData>& shapeData, std::vector<glm::vec3>& vertices, int material_a, int material_b)
{
	std::vector<int> v_start_id;
	std::vector<int> tri_end_id;
	std::vector<int> material_id;

	std::vector<glm::ivec3> triangles;
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

	BufferData<glm::vec3>(m_Scene->vertices, vertices.data(), vertices.size());

	for (int i = 0, count = 0, obj = 0; i < triangles.size(); ++i, ++count)
	{
		if (count == tri_end_id[obj])
		{
			count = 0;
			++obj;
		}
		shapeData.emplace_back(material_id[obj], triangles[i], m_Scene->vertices);
	}
}

void TestLayer::AddTwoBox_Triangles(std::vector<ShapeData>& shapeData, std::vector<glm::vec3>& vertices, int material_a, int material_b)
{
	std::vector<int> v_start_id;
	std::vector<int> tri_end_id;
	std::vector<int> material_id;

	std::vector<glm::ivec3> triangles;
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

	Shape::ComputeTransform(glm::vec3(0, 3, -2), glm::vec3(0, 0, 0), glm::vec3(1, 1, 1), transform);

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
	
	Shape::ComputeTransform(glm::vec3(0, 3, 2), glm::vec3(0, 0, 0), glm::vec3(1, 1, 1), transform);
	
	for (int i = v_start_id.back(); i < vertices.size(); ++i)
	{
		vertices[i] = glm::vec3(transform * glm::vec4(vertices[i], 1.f));
	}

	BufferData<glm::vec3>(m_Scene->vertices, vertices.data(), vertices.size());

	for (int i = 0, count = 0, obj = 0; i < triangles.size(); ++i, ++count)
	{
		if (count == tri_end_id[obj])
		{
			count = 0;
			++obj;
		}
		shapeData.emplace_back(material_id[obj], triangles[i], m_Scene->vertices);
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
	const char* albedo_tex_path = "E://Projects//CUDA_Projects//CudaPBRT//res//textures//brick.jpg";
	const char* nor_tex_path = "E://Projects//CUDA_Projects//CudaPBRT//res//textures//brickN.jpg";
	const char* env_map_path = "E://Projects//CUDA_Projects//CudaPBRT//res//environment_maps//fireplace_4k.hdr";

	m_Textures.emplace_back(CudaTexture::CreateCudaTexture(albedo_tex_path));
	m_Textures.emplace_back(CudaTexture::CreateCudaTexture(nor_tex_path));
	m_Textures.emplace_back(CudaTexture::CreateCudaTexture(env_map_path));

	// Hard-coding Cornell Box Scene

	// material data
	int matteWhiteId	= 0;
	int matteRedId		= 1;
	int matteGreenId	= 2;
	int mirrorId		= 3;
	int glassId			= 4;
	int tex_Id			= 5;

	std::vector<MaterialData> materialData;
	materialData.emplace_back(MaterialType::DiffuseReflection, glm::vec3(0.85, 0.81, 0.78)); //matteWhite
	materialData.emplace_back(MaterialType::DiffuseReflection, glm::vec3(0.63, 0.065, 0.05)); //matteRed
	materialData.emplace_back(MaterialType::DiffuseReflection, glm::vec3(0.14, 0.45, 0.091)); //matteGreen
	materialData.emplace_back(MaterialType::SpecularReflection, glm::vec3(1.f, 1.f, 1.f)); // mirror
	materialData.emplace_back(MaterialType::SpecularTransmission, glm::vec3(.9f, .9f, 1.f), 0.f, 0.f, 1.55f); // glass
	//materialData.emplace_back(MaterialType::DiffuseReflection, m_Textures[0]->GetTextureObject()); // texture

	// shape data
	std::vector<ShapeData> shapeData;

	shapeData.emplace_back(ShapeType::Square, matteWhiteId, glm::vec3(0, -2.5, 0), glm::vec3(-90, 0, 0), glm::vec3(10, 10, 1)); // Floor
	shapeData.emplace_back(ShapeType::Square, matteRedId, glm::vec3(5, 2.5, 0), glm::vec3(0, -90, 0), glm::vec3(10, 10, 1)); // Red wall
	shapeData.emplace_back(ShapeType::Square, matteGreenId, glm::vec3(-5, 2.5, 0), glm::vec3(0, 90, 0), glm::vec3(10, 10, 1)); // Green Wall
	shapeData.emplace_back(ShapeType::Square, matteWhiteId, glm::vec3(0, 2.5, 5), glm::vec3(0, 180, 0), glm::vec3(10, 10, 1)); // Back Wall
	shapeData.emplace_back(ShapeType::Square, matteWhiteId, glm::vec3(0, 7.5, 0), glm::vec3(90, 0, 0), glm::vec3(10, 10, 1)); // Ceiling

	std::vector<glm::vec3> vertices;

	//TestSingleTriangle(shapeData, vertices);
	AddCornellBox_Triangles(shapeData, vertices, glassId, matteWhiteId);
	//AddTwoBox_Triangles(shapeData, vertices, matteWhiteId, matteWhiteId);

	//shapeData.emplace_back(ShapeType::Sphere, glassId, glm::vec3(0, 1.25, 0), glm::vec3(0, 0, 0), glm::vec3(3, 3, 3));
	//shapeData.emplace_back(ShapeType::Cube, glassId, glm::vec3(2, 0, 3), glm::vec3(0, 27.5, 0), glm::vec3(3, 6, 3)); // Long Cube
	//shapeData.emplace_back(ShapeType::Cube, matteWhiteId, glm::vec3(-2, -1, 0.75), glm::vec3(0, -17.5, 0), glm::vec3(3, 3, 3)); // Short Cube
	//LoadObj(shapeData, vertices, "E://Projects//CUDA_Projects//CudaPBRT//res//models//sphere.obj");
	CreateBoundingBox(shapeData, vertices);
	// Light
	std::vector<LightData> lightData;
	ShapeData areaLightShape(ShapeType::Square, -1, glm::vec3(0, 7.45, 0), glm::vec3(90, 0, 0), glm::vec3(3, 3, 1));
	Spectrum Le(40);
	lightData.emplace_back(LightType::ShapeLight, areaLightShape, Le);

	CreateArrayOnCude<Shape, ShapeData>(m_Scene->shapes, m_Scene->shape_count, shapeData);
	CreateArrayOnCude<Material, MaterialData>(m_Scene->materials, m_Scene->material_count, materialData);
	CreateArrayOnCude<Light, LightData>(m_Scene->lights, m_Scene->light_count, lightData);
}

void TestLayer::LoadObj(std::vector<ShapeData>& shapeData, std::vector<glm::vec3>& vertices, const char* path)
{
	std::ifstream in(path, std::ios::in);

	int v_start_id = vertices.size();
	std::vector<glm::ivec3> triangles;

	int line_count = 0;
	int f_start = 0;
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

		}
		else if (line.starts_with("vt"))
		{

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
			
			glm::ivec3 v_id;

			for (int i = 0; i < 3; ++i)
			{
				auto id_str = result[i + 1] | std::views::split('/') | std::views::transform([](auto word) {
					return std::string(word.begin(), word.end());
				});
				std::vector<std::string> ids(id_str.begin(), id_str.end());
				v_id[i] = std::stoi(ids[0]) - f_start;
			}

			triangles.emplace_back(v_id);
		}
	}

	glm::mat4 transform;
	Shape::ComputeTransform(glm::vec3(0, 1.25, 0), glm::vec3(0, 0, 0), glm::vec3(2, 2, 2), transform);

	for (int i = v_start_id; i < vertices.size(); ++i)
	{
		vertices[i] = glm::vec3(transform * glm::vec4(vertices[i], 1.f));
	}

	BufferData<glm::vec3>(m_Scene->vertices, vertices.data(), vertices.size());

	for (int i = 0; i < triangles.size(); ++i)
	{
		shapeData.emplace_back(4, triangles[i], m_Scene->vertices);
	}
}
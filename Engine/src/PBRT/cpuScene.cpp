#include "cpuScene.h"

#include "MeshLoader/meshLoader.h"

#define SafeGet(attr, json, str, type) if(json.contains(str)) { attr = json[str].get<type>(); }
#define SafeGetVec3(attr, json, str) if(json.contains(str)) \
									{ JSON arr = json[str]; attr = glm::vec3(arr[0].get<float>(), arr[1].get<float>(), arr[2].get<float>()); }

namespace CudaPBRT
{
	CPUScene::CPUScene()
	{
		camera = mkU<PerspectiveCamera>(680, 680, 19.5f, glm::vec3(0, 5.5, -30), glm::vec3(0, 2.5, 0));
		camController = mkU<PerspectiveCameraController>(*camera);
	}

	CPUScene::CPUScene(const char* path)
	{
		camera = mkU<PerspectiveCamera>(680, 680, 19.5f, glm::vec3(0, 5.5, -30), glm::vec3(0, 2.5, 0));
		camController = mkU<PerspectiveCameraController>(*camera);

		LoadSceneFromJsonFile(path);
	}

	bool CPUScene::LoadCameraFromJSON(const JSON& json_data)
	{
		SafeGetVec3(camera->ref, json_data, "ref");
		SafeGetVec3(camera->position, json_data, "pos");
		SafeGetVec3(camera->up, json_data, "up");

		SafeGet(camera->fovy, json_data, "fovy", float);
		SafeGet(camera->focalDistance, json_data, "focalDistance", float);
		SafeGet(camera->lensRadius, json_data, "lensRadius", unsigned int);
		SafeGet(camera->width, json_data, "width", unsigned int);
		SafeGet(camera->height, json_data, "height", unsigned int);

		camera->RecomputeAttributes();

		return true;
	}

	bool CPUScene::LoadMaterialFromJSON(const JSON& material)
	{
		std::string type_str;
		SafeGet(type_str, material, "type", std::string);
		MaterialType type = Str2MaterialType(type_str.c_str());
		if (type == MaterialType::None)
		{
			return false;
		}

		std::string name;
		SafeGet(name, material, "name", std::string);

		// read albedo, roughness, metallic, eta
		glm::vec3 albedo(0.f);
		SafeGetVec3(albedo, material, "albedo");

		float roughness = 0.5f, metallic = 0.5f, eta = AirETA;

		SafeGet(roughness, material, "roughness", float);
		SafeGet(metallic, material, "metallic", float);
		SafeGet(eta, material, "eta", float);

		materialData.emplace_back(type, albedo, roughness, metallic, eta); //matteRed
		materials_map.emplace(name, materialData.size() - 1);

		MaterialData& material_data = materialData.back();

		// try to read textures
		if (material.contains("albedo_map"))
		{
			JSON texture_data = material["albedo_map"];

			std::string albedo_map_path;
			SafeGet(albedo_map_path, texture_data, "path", std::string);

			bool flip_v = false;
			SafeGet(flip_v, texture_data, "flip_v", bool);

			m_Textures.emplace_back(CudaTexture::CreateCudaTexture(albedo_map_path.c_str(), flip_v));
			material_data.albedoMapId = m_Textures.back()->GetTextureObject();
		}

		if (material.contains("normal_map"))
		{
			JSON texture_data = material["normal_map"];

			std::string normal_map_path;
			SafeGet(normal_map_path, texture_data, "path", std::string);

			bool flip_v = false;
			SafeGet(flip_v, texture_data, "flip_v", bool);

			m_Textures.emplace_back(CudaTexture::CreateCudaTexture(normal_map_path.c_str(), flip_v));
			material_data.normalMapId = m_Textures.back()->GetTextureObject();
		}

		if (material.contains("roughness_map"))
		{
			JSON texture_data = material["roughness_map"];

			std::string roughness_map_path;
			SafeGet(roughness_map_path, texture_data, "path", std::string);

			bool flip_v = false;
			SafeGet(flip_v, texture_data, "flip_v", bool);

			m_Textures.emplace_back(CudaTexture::CreateCudaTexture(roughness_map_path.c_str(), flip_v));
			material_data.roughnessMapId = m_Textures.back()->GetTextureObject();
		}

		if (material.contains("roughness_map"))
		{
			JSON texture_data = material["metallic_map"];

			std::string metallic_map_path;
			SafeGet(metallic_map_path, texture_data, "path", std::string);

			bool flip_v = false;
			SafeGet(flip_v, texture_data, "flip_v", bool);

			m_Textures.emplace_back(CudaTexture::CreateCudaTexture(metallic_map_path.c_str(), flip_v));
			material_data.metallicMapId = m_Textures.back()->GetTextureObject();
		}
		return true;
	}

	bool CPUScene::LoadShapeFromJSON(const JSON& object)
	{
		static const int DefaultMaterialId = 0;

		std::string type_str;
		SafeGet(type_str, object, "type", std::string);

		if (type_str == "obj")
		{
			std::string path;
			SafeGet(path, object, "path", std::string);

			ObjectData obj_data;
			SafeGetVec3(obj_data.transform.translation, object, "translation");
			SafeGetVec3(obj_data.transform.rotation, object, "rotation");
			SafeGetVec3(obj_data.transform.scale, object, "scale");

			std::string material_name;
			SafeGet(material_name, object, "material", std::string);
			obj_data.material_id = DefaultMaterialId;
			
			if (materials_map.find(material_name) != materials_map.end())
			{
				obj_data.material_id = materials_map[material_name];
			}

			LoadMeshFromFile(path.c_str(), obj_data);

			objectData.push_back(obj_data);
		}
		else
		{
			ShapeType type = Str2ShapeType(type_str.c_str());
			
			if (type == ShapeType::None)
			{
				return false;
			}

			int material_id = DefaultMaterialId;;
			std::string material_name;
			SafeGet(material_name, object, "material", std::string);
			if (materials_map.find(material_name) != materials_map.end())
			{
				material_id = materials_map[material_name];
			}
			
			glm::vec3 translation(0.f), rotation(0.f), scale(1.f);
			SafeGetVec3(translation, object, "translation");
			SafeGetVec3(rotation, object, "rotation");
			SafeGetVec3(scale, object, "scale");

			shapeData.emplace_back(type, material_id, translation, rotation, scale);
		}
		return true;
	}

	bool CPUScene::LoadLightFromJSON(const JSON& light)
	{
		std::string type_str;
		SafeGet(type_str, light, "type", std::string);

		LightType type = Str2LightType(type_str.c_str());

		switch(type)
		{
		case LightType::ShapeLight:
		{
			std::string shape_str;
			SafeGet(shape_str, light, "shape", std::string);
			ShapeType shape_type = Str2ShapeType(shape_str.c_str());
			
			glm::vec3 translation(0.f), rotation(0.f), scale(1.f);
			SafeGetVec3(translation, light, "translation");
			SafeGetVec3(rotation, light, "rotation");
			SafeGetVec3(scale, light, "scale");

			ShapeData light_shape(shape_type, -1, translation, rotation, scale);

			float Le;
			SafeGet(Le, light, "Le", float);

			lightData.emplace_back(type, light_shape, Spectrum(Le));

			return true;
		}
		default:
			return false;
		}

		return true;
	}

	bool CPUScene::LoadSceneFromJsonFile(const char* path)
	{
		// clear previous scene data
		ClearScene();

		// try to open json file
		std::ifstream in(path);
		if (!in.is_open())
		{
			printf("Cannot open %s\n!", path);
			return false;
		}
		JSON json_data = JSON::parse(in);

		LoadCameraFromJSON(json_data["camera"]);

		JSON scene_data = json_data["scene"];

		// load materials
		// default material
		materialData.emplace_back(MaterialType::LambertianReflection, glm::vec3(0.85, 0.81, 0.78)); //matteWhite, default material
		materials_map.emplace("default", 0);
		int DefaultMaterialId = 0;

		if (scene_data.contains("materials"))
		{
			JSON materials_list = scene_data["materials"];
			for (const auto& material : materials_list)
			{
				LoadMaterialFromJSON(material);
			}
		}

		// load environment map
		if (scene_data.contains("environment map"))
		{
			std::string env_map_path;
			SafeGet(env_map_path, scene_data["environment map"], "path", std::string);

			m_Textures.emplace_back(CudaTexture::CreateCudaTexture(env_map_path.c_str(), true));
			m_GPUScene.envMap.SetTexObj(m_Textures.back()->GetTextureObject());
		}

		if (scene_data.contains("objects"))
		{
			JSON objects_list = scene_data["objects"];
			for (const auto& object : objects_list)
			{
				LoadShapeFromJSON(object);
			}
		}

		// buffer shape datas
		BufferData<glm::vec3>(m_GPUScene.vertices, vertices.data(), vertices.size());
		BufferData<glm::vec3>(m_GPUScene.normals, normals.data(), normals.size());
		BufferData<glm::vec2>(m_GPUScene.uvs, uvs.data(), uvs.size());

		for (const ObjectData& data : objectData)
		{
			for (int i = data.start_id; i < data.end_id; ++i)
			{
				shapeData.emplace_back(data.material_id, triangles[i], m_GPUScene.vertices, m_GPUScene.normals, m_GPUScene.uvs);
			}
		}

		// shapeData's order will be changed 
		CreateBoundingBox(shapeData, vertices);

		// load lights
		if (scene_data.contains("lights"))
		{
			JSON lights_list = scene_data["lights"];
			for (const auto& light : lights_list)
			{
				LoadLightFromJSON(light);
			}
		}

		// create shapes, meterials, and lights on cuda
		CreateArrayOnCude<Shape, ShapeData>(m_GPUScene.shapes, m_GPUScene.shape_count, shapeData);
		CreateArrayOnCude<Material, MaterialData>(m_GPUScene.materials, m_GPUScene.material_count, materialData);
		CreateArrayOnCude<Light, LightData>(m_GPUScene.lights, m_GPUScene.light_count, lightData);

		return true;
	}

	bool CPUScene::LoadMeshFromFile(const char* path, ObjectData& obj_data)
	{
		// try to open file
		uPtr<MeshLoader> mesh_loader = MeshLoader::CreateMeshLoad(path);

		if (mesh_loader)
		{
			obj_data.start_id = triangles.size();

			size_t v_start_id = vertices.size();
			size_t n_start_id = normals.size();

			// load mesh data
			mesh_loader->Load(triangles, vertices, normals, uvs);

			// apply transformation
			glm::mat4 trans;
			glm::mat3 transposeInvT;
			Shape::ComputeTransform(obj_data.transform.translation, 
									obj_data.transform.rotation, 
									obj_data.transform.scale, trans);
			transposeInvT = glm::transpose(glm::inverse(glm::mat3(trans)));

			// TODO: accelerate by CUDA
			for (size_t i = v_start_id; i < vertices.size(); ++i)
			{
				vertices[i] = glm::vec3(trans * glm::vec4(vertices[i], 1.f));
			}

			for (size_t i = n_start_id; i < normals.size(); ++i)
			{
				normals[i] = glm::normalize(transposeInvT * normals[i]);
			}

			obj_data.end_id = triangles.size();
			return true;
		}
		return false;
	}

	void CPUScene::CreateBoundingBox(std::vector<ShapeData>& shapeData, std::vector<glm::vec3>& vertices)
	{
		std::vector<BoundingBox> boundingBoxes;
		std::vector<BVHNode> bvh_nodes;

		CreateBVH(shapeData, vertices, boundingBoxes, bvh_nodes);

		BufferData<BoundingBox>(m_GPUScene.boundings, boundingBoxes.data(), boundingBoxes.size());
		BufferData<BVHNode>(m_GPUScene.BVH, bvh_nodes.data(), bvh_nodes.size());
	}
}
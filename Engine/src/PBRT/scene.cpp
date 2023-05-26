#include "scene.h"

#include "MeshLoader/meshLoader.h"

namespace CudaPBRT
{
	CPUScene::CPUScene(const char* path)
	{
		LoadSceneFromJSON(path);
	}

	void CPUScene::LoadSceneFromJSON(const char* path)
	{
		// clear previous scene data
		ClearScene();

		// try to open json file

		// load materials
		materialData.emplace_back(MaterialType::LambertianReflection, glm::vec3(0.85, 0.81, 0.78)); //matteWhite, default material
		materialData.emplace_back(MaterialType::LambertianReflection, glm::vec3(0.63, 0.065, 0.05)); //matteRed
		materialData.emplace_back(MaterialType::LambertianReflection, glm::vec3(0.14, 0.45, 0.091)); //matteGreen
		materialData.emplace_back(MaterialType::SpecularReflection, glm::vec3(1.f, 1.f, 1.f)); // mirror
		materialData.emplace_back(MaterialType::SpecularTransmission, glm::vec3(.9f, .9f, 1.f), 0.f, 0.f, 1.55f); // glass

		const char* cam_albedo_path = "E://Projects//CUDA_Projects//CudaPBRT//res//textures//#CAM0001_Textures_COL_2k.png";
		const char* cam_nor_path = "E://Projects//CUDA_Projects//CudaPBRT//res//textures//#CAM0001_Textures_NRML_2k.png";
		const char* cam_metl_path = "E://Projects//CUDA_Projects//CudaPBRT//res//textures//#CAM0001_Textures_METL_2k.png";
		const char* cam_rougn_path = "E://Projects//CUDA_Projects//CudaPBRT//res//textures//#CAM0001_Textures_ROUGH_2k.png";
		m_Textures.emplace_back(CudaTexture::CreateCudaTexture(cam_albedo_path, true));
		m_Textures.emplace_back(CudaTexture::CreateCudaTexture(cam_nor_path, true));
		m_Textures.emplace_back(CudaTexture::CreateCudaTexture(cam_metl_path, true));
		m_Textures.emplace_back(CudaTexture::CreateCudaTexture(cam_rougn_path, true));

		
		materialData.emplace_back(MaterialType::MetallicWorkflow, m_Textures[0]->GetTextureObject(),
								  m_Textures[1]->GetTextureObject(),
								  m_Textures[2]->GetTextureObject(),
								  m_Textures[3]->GetTextureObject()); // texture MetallicWorkflow
		
		materialData.emplace_back(MaterialType::MicrofacetReflection, glm::vec3(.8f, .8f, .8f), .5f, 0.5f); // microfacet
		materialData.emplace_back(MaterialType::MetallicWorkflow, glm::vec3(.8f, .8f, .8f), 1.0f, 0.0f); // MetallicWorkflow

		// load environment map
		const char* env_map_path = "E://Projects//CUDA_Projects//CudaPBRT//res//environment_maps//interior_atelier_soft_daylight.hdr";
		m_Textures.emplace_back(CudaTexture::CreateCudaTexture(env_map_path, true));
		m_GPUScene.envMap.SetTexObj(m_Textures.back()->GetTextureObject());

		int DefaultMaterialId = 0;
		int matteRedId = 1;
		int matteGreenId = 2;
		int mirrorId = 3;
		int glassId = 4;
		int obj_tex_Id = 5;
		int microfacetId = 6;
		int MetallicWfId = 7;
		
		// load meshes
		ObjectData obj_data;
		obj_data.transform.scale = glm::vec3(.65f);
		obj_data.transform.rotation.y = 180.f;
		obj_data.material_id = obj_tex_Id;
		LoadObj("E://Projects//CUDA_Projects//CudaPBRT//res//models//Camera.obj", obj_data);
		objectData.push_back(obj_data);

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
		ShapeData areaLightShape(ShapeType::Square, -1, glm::vec3(0, 7.45, 0), glm::vec3(90, 0, 0), glm::vec3(3, 3, 1));
		Spectrum Le(40);
		//lightData.emplace_back(LightType::ShapeLight, areaLightShape, Le);
		
		// create shapes, meterials, and lights on cuda
		CreateArrayOnCude<Shape, ShapeData>(m_GPUScene.shapes, m_GPUScene.shape_count, shapeData);
		CreateArrayOnCude<Material, MaterialData>(m_GPUScene.materials, m_GPUScene.material_count, materialData);
		CreateArrayOnCude<Light, LightData>(m_GPUScene.lights, m_GPUScene.light_count, lightData);
	}

	void CPUScene::LoadObj(const char* path, ObjectData& obj_data)
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
		}
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
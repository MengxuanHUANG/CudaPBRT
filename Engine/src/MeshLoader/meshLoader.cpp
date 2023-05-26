#include "meshLoader.h"

#include <filesystem>

#include "Core/Utilities.h"

#include "PBRT/Shape/shape.h"

namespace CudaPBRT
{
	uPtr<MeshLoader> MeshLoader::CreateMeshLoad(const char* path)
	{
		std::filesystem::path file_path(path);
		std::ifstream fin(path);
		if (!fin.is_open())
		{
			printf("Cannot open %s!", path);
			return nullptr;
		}

		if (file_path.extension() == ".obj")
		{
			return mkU<ObjMeshLoader>(path);
		}
		else
		{
			return nullptr;
		}
	}

	ObjMeshLoader::ObjMeshLoader(const char* path)
		: in(path)
	{}

	void ObjMeshLoader::Load(std::vector<TriangleData>& triangles,
							 std::vector<glm::vec3>& vertices,
							 std::vector<glm::vec3>& normals,
							 std::vector<glm::vec2>& uvs)
	{
		const static int f_start = 1;

		for (std::string line; std::getline(in, line);)
		{
			if (line.starts_with("vn"))
			{
				std::vector<std::string> result = StringUtility::Split(line, " ");

				normals.emplace_back(std::stof(result[1]), std::stof(result[2]), std::stof(result[3]));
			}
			else if (line.starts_with("vt"))
			{
				std::vector<std::string> result = StringUtility::Split(line, " ");

				uvs.emplace_back(std::stof(result[1]), std::stof(result[2]));
			}
			else if (line.starts_with("v"))
			{
				std::vector<std::string> result = StringUtility::Split(line, " ");

				vertices.emplace_back(std::stof(result[1]), std::stof(result[2]), std::stof(result[3]));
			}
			else if (line.starts_with("f"))
			{
				std::vector<std::string> result = StringUtility::Split(line, " ");

				ASSERT(result.size() > 3);

				std::vector<int> v_id;
				std::vector<int> n_id;
				std::vector<int> uv_id;

				v_id.resize(result.size() - 1);
				n_id.resize(result.size() - 1);
				uv_id.resize(result.size() - 1);

				for (int i = 0; i < result.size() - 1; ++i)
				{
					std::vector<std::string> ids = StringUtility::Split(result[i + 1], "/");
					v_id[i]		= std::stoi(ids[0]) - f_start;
					uv_id[i]	= std::stoi(ids[1]) - f_start;
					n_id[i]		= std::stoi(ids[2]) - f_start;
				}
				// naive triangulation
				for (int i = 1; i < result.size() - 2; ++i)
				{
					triangles.emplace_back(glm::ivec3(v_id[0], v_id[i], v_id[i + 1]),
										   glm::ivec3(n_id[0], n_id[i], n_id[i + 1]),
										   glm::ivec3(uv_id[0], uv_id[i], uv_id[i + 1]));
				}
			}
		}
	}
}
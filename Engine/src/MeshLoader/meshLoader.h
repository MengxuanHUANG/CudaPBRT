#pragma once

#include "Core/Core.h"
#include <fstream>
#include <vector>
#include <glm/glm.hpp>

namespace CudaPBRT
{
	struct TriangleData;

	class MeshLoader
	{
	public:
		virtual ~MeshLoader() = default;

		virtual void Load(std::vector<TriangleData>& triangles,
						  std::vector<glm::vec3>& vertices,
						  std::vector<glm::vec3>& normals,
						  std::vector<glm::vec2>& uvs) = 0;
	
	public:
		static uPtr<MeshLoader> CreateMeshLoad(const char* path);
	};

	class ObjMeshLoader : public MeshLoader
	{
	public:
		ObjMeshLoader(const char* path);
		virtual ~ObjMeshLoader() = default;

		virtual void Load(std::vector<TriangleData>& triangles,
						  std::vector<glm::vec3>& vertices,
						  std::vector<glm::vec3>& normals,
						  std::vector<glm::vec2>& uvs) override;
	protected:
		std::ifstream in;
	};
}
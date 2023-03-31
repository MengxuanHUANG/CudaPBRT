#pragma once
#include "PBRT/pbrtDefine.h"

#include <glm/glm.hpp>

namespace CudaPBRT
{
	struct MaterialData
	{
		glm::vec3 albedo = glm::vec3(1.f); // default color is white
		float metallic = 0.5f;
		float roughness = 0.5f;

		float eta = 1.000293f; // IOR of air

		int albedoMapId		= -1;
		int normalMapId		= -1;
		int metallicMapId	= -1;
		int roughnessMapId	= -1;
	};

	// GPU side object
	class Material
	{
	public:
		CUDA_ONLY Material(MaterialData& mData)
			:m_MaterialData(mData)
		{}

		CUDA_ONLY virtual ~Material() = default;

		CUDA_ONLY virtual glm::vec3 GetAlbedo(const glm::vec2& uv) const = 0;
		CUDA_ONLY virtual glm::vec3 GetNormal(const glm::vec2& uv) const = 0;
		CUDA_ONLY virtual float GetMetallic(const glm::vec2& uv) const = 0;
		CUDA_ONLY virtual float GetRoughness(const glm::vec2& uv) const = 0;
	protected:
		MaterialData& m_MaterialData;
	};
}

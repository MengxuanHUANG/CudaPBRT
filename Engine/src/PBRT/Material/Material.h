#pragma once
#include "PBRT/pbrtDefine.h"
#include "PBRT/bsdf.h"

namespace CudaPBRT
{
	enum class MaterialType : unsigned char
	{
		Specular = BIT(7),
		None = 0,
		DiffuseReflection		= 1U,
		SpecularReflection		= 2U | Specular,
		SpecularTransmission	= 3U | Specular
	};

	INLINE CPU_GPU bool MaterialIs(MaterialType type, MaterialType flag)
	{
		return static_cast<int>(type) & static_cast<int>(flag);
	}

	struct MaterialData
	{
		MaterialType type = MaterialType::None;
		glm::vec3 albedo = glm::vec3(1.f); // default color is white
		float metallic = 0.5f;
		float roughness = 0.5f;

		float eta = AirETA; // IOR of air

		int albedoMapId		= -1;
		int normalMapId		= -1;
		int metallicMapId	= -1;
		int roughnessMapId	= -1;

		MaterialData(MaterialType type, 
					 const glm::vec3& albedo = glm::vec3(1.f),
					 float metallic = 0.5f,
					 float roughness = 0.5f,
					 float eta = 1.000293f,
					 int albedoMapId = -1,
					 int normalMapId = -1,
					 int metallicMapId = -1,
					 int roughnessMapId = -1)
			: type(type), 
			  albedo(albedo), 
			  metallic(metallic), 
			  roughness(roughness), 
			  eta(eta), 
			  albedoMapId(albedoMapId), 
			  normalMapId(normalMapId), 
			  metallicMapId(metallicMapId), 
			  roughnessMapId(roughnessMapId)
		{}
	};

	// GPU side object
	class Material
	{
	public:
		CPU_GPU Material(const MaterialData& mData, BxDF* bxdf)
			:m_MaterialData(mData), m_BSDF(bxdf)
		{}

		CPU_GPU virtual ~Material() {}

		CPU_GPU BSDF& GetBSDF() { return m_BSDF; }

		CPU_GPU Spectrum GetAlbedo(const glm::vec2& uv = glm::vec2(0, 0)) const
		{
			return Spectrum(m_MaterialData.albedo);
		}

		CPU_GPU glm::vec3 GetNormal(const glm::vec3& normal, const glm::vec2& uv = glm::vec2(0, 0)) const
		{
			return normal;
		}

		CPU_GPU float GetMetallic(const glm::vec2& uv = glm::vec2(0, 0)) const
		{
			return m_MaterialData.metallic;
		}

		CPU_GPU float GetRoughness(const glm::vec2& uv = glm::vec2(0, 0)) const
		{
			return m_MaterialData.roughness;
		}

	public:
		MaterialData m_MaterialData;
		BSDF m_BSDF;
	};
}

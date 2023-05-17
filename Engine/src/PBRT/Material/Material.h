#pragma once
#include "PBRT/pbrtDefine.h"
#include "materialUtilities.h"

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

		CudaTexObj albedoMapId		= 0;
		CudaTexObj normalMapId		= 0;
		CudaTexObj metallicMapId	= 0;
		CudaTexObj roughnessMapId	= 0;

		MaterialData(MaterialType type, 
					 const glm::vec3& albedo = glm::vec3(1.f),
					 float metallic = 0.5f,
					 float roughness = 0.5f,
					 float eta = AirETA)
			: type(type), 
			  albedo(albedo), 
			  metallic(metallic), 
			  roughness(roughness), 
			  eta(eta), 
			  albedoMapId(0), 
			  normalMapId(0), 
			  metallicMapId(0), 
			  roughnessMapId(0)
		{}

		MaterialData(MaterialType type,
					 CudaTexObj albedo_map_id,
					 CudaTexObj normal_map_id = 0,
					 CudaTexObj metallic_map_id = 0,
					 CudaTexObj roughness_map_id = 0,
					 float eta = AirETA)
			: type(type),
			  albedo(glm::vec3(1.f)),
			  metallic(0.f),
			  roughness(0.f),
			  eta(eta),
			  albedoMapId(albedo_map_id),
			  normalMapId(normal_map_id),
			  metallicMapId(metallic_map_id),
			  roughnessMapId(roughness_map_id)
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

		INLINE CPU_GPU BSDF& GetBSDF() { return m_BSDF; }

		INLINE GPU_ONLY Spectrum GetAlbedo(const glm::vec2& uv = glm::vec2(0, 0)) const
		{
			if (m_MaterialData.albedoMapId > 0)
			{
				float4 color = ReadTexture(m_MaterialData.albedoMapId, uv);
				glm::vec3 albedo(color.x, color.y, color.z);
				//albedo = glm::pow(albedo, glm::vec3(2.2f));
				return Spectrum(albedo);
			}
			else
			{
				return Spectrum(m_MaterialData.albedo);
			}
		}

		INLINE GPU_ONLY glm::vec3 GetNormal(const glm::vec3& normal, const glm::vec2& uv = glm::vec2(0, 0)) const
		{
			if (m_MaterialData.normalMapId > 0)
			{
				// TODO: recompute normal
				float4 nor = ReadTexture(m_MaterialData.normalMapId, uv);
				glm::vec3 tan, bitan;

				coordinateSystem(normal, tan, bitan);

				return glm::normalize(nor.x * tan + nor.y * bitan + nor.z * normal);
			}
			else
			{
				return normal;
			}
		}

		INLINE GPU_ONLY float GetMetallic(const glm::vec2& uv = glm::vec2(0, 0)) const
		{
			if (m_MaterialData.metallicMapId > 0)
			{
				float4 metallic = ReadTexture(m_MaterialData.metallicMapId, uv);
				return m_MaterialData.metallic;
			}
			else
			{
				return m_MaterialData.metallic;
			}
		}

		INLINE GPU_ONLY float GetRoughness(const glm::vec2& uv = glm::vec2(0, 0)) const
		{
			if (m_MaterialData.roughnessMapId > 0)
			{
				float4 roughness = ReadTexture(m_MaterialData.roughnessMapId, uv);
				return m_MaterialData.roughness;
			}
			else
			{
				return m_MaterialData.roughness;
			}
		}

	public:
		MaterialData m_MaterialData;
		BSDF m_BSDF;
	};
}

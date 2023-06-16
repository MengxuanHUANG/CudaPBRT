#pragma once
#include "materialUtilities.h"
#include "PBRT/Shape/shape.h"

#include "bsdf.h"

namespace CudaPBRT
{
	enum class MaterialType : unsigned char
	{
		Specular = BIT(7),
		None = 0,
		LambertianReflection	= 1U,
		SpecularReflection		= 2U | Specular,
		SpecularTransmission	= 3U | Specular,
		Glass					= 4U | Specular,
		MicrofacetReflection	= 5U,
		MetallicWorkflow		= 6U
	};
	
	MaterialType Str2MaterialType(const char* str);

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
		
		float Lv = 0.f;

		float eta = AirETA; // IOR of air

		CudaTexObj albedoMapId		= 0;
		CudaTexObj normalMapId		= 0;
		CudaTexObj metallicMapId	= 0;
		CudaTexObj roughnessMapId	= 0;
		CudaTexObj LvMapId			= 0;

		bool lightMaterial			= false;

		MaterialData(){}

		MaterialData(MaterialType type, 
					 const glm::vec3& albedo = glm::vec3(1.f),
					 float metallic = 0.5f,
					 float roughness = 0.5f,
					 float Lv = 0.f,
					 float eta = AirETA,
					 bool light_material = false)
			: type(type), 
			  albedo(albedo), 
			  metallic(metallic), 
			  roughness(roughness),
			  Lv(Lv),
			  eta(eta), 
			  albedoMapId(0), 
			  normalMapId(0), 
			  metallicMapId(0), 
			  roughnessMapId(0),
			  LvMapId(0),
			  lightMaterial(light_material)
		{}

		MaterialData(MaterialType type,
					 CudaTexObj albedo_map_id,
					 CudaTexObj normal_map_id = 0,
					 CudaTexObj metallic_map_id = 0,
					 CudaTexObj roughness_map_id = 0,
					 CudaTexObj Lv_map_id = 0,
					 float eta = AirETA,
					 bool light_material = false)
			: type(type),
			  albedo(glm::vec3(1.f)),
			  metallic(0.f),
			  roughness(0.f),
			  eta(eta),
			  albedoMapId(albedo_map_id),
			  normalMapId(normal_map_id),
			  metallicMapId(metallic_map_id),
			  roughnessMapId(roughness_map_id),
			  LvMapId(Lv_map_id),
			  lightMaterial(light_material)
		{}
	};

	// GPU side object
	class Material
	{
	public:
		CPU_GPU Material()
		{}

		CPU_GPU Material(const MaterialData& mData)
			:m_MaterialData(mData)
		{}

		INLINE CPU_GPU const BSDF& GetBSDF() const { return m_BSDF; }

		INLINE GPU_ONLY Spectrum GetAlbedo(const glm::vec2& uv = glm::vec2(0, 0)) const
		{
			return (m_MaterialData.albedoMapId > 0 ? GetAlbedoMap(uv) : Spectrum(m_MaterialData.albedo));
		}

		INLINE GPU_ONLY glm::vec3 GetNormal(const glm::vec3& normal, const glm::vec2& uv = glm::vec2(0, 0)) const
		{
			return (m_MaterialData.normalMapId > 0 ? GetNormalMap(normal, uv) : normal);
		}

		INLINE GPU_ONLY float GetMetallic(const glm::vec2& uv = glm::vec2(0, 0)) const
		{
			return (m_MaterialData.metallicMapId > 0 ? ReadTexture(m_MaterialData.metallicMapId, uv).x : m_MaterialData.metallic);
		}

		INLINE GPU_ONLY float GetRoughness(const glm::vec2& uv = glm::vec2(0, 0)) const
		{
			return (m_MaterialData.roughnessMapId > 0 ? ReadTexture(m_MaterialData.roughnessMapId, uv).x : m_MaterialData.roughness);
		}

		INLINE GPU_ONLY float GetLv(const glm::vec2& uv = glm::vec2(0, 0)) const
		{
			return (m_MaterialData.LvMapId > 0 ? ReadTexture(m_MaterialData.LvMapId, uv).x : m_MaterialData.Lv);
		}

		INLINE GPU_ONLY Spectrum GetIrradiance(const glm::vec2& uv = glm::vec2(0, 0)) const
		{
			return GetLv(uv) * GetAlbedo(uv);
		}

		INLINE GPU_ONLY Spectrum GetAlbedo(const glm::vec3& p, const Shape& shape) const
		{
			return (m_MaterialData.albedoMapId > 0 ? GetAlbedoMap(shape.GetUV(p)) : Spectrum(m_MaterialData.albedo));
		}

		INLINE GPU_ONLY float GetLv(const glm::vec3& p, const Shape& shape) const
		{
			return (m_MaterialData.LvMapId > 0 ? ReadTexture(m_MaterialData.LvMapId, shape.GetUV(p)).x : m_MaterialData.Lv);
		}

		INLINE GPU_ONLY Spectrum GetIrradiance(const glm::vec3& p, const Shape& shape) const
		{
			return GetLv(p, shape) * GetAlbedo(p, shape);
		}

	protected:
		INLINE GPU_ONLY Spectrum GetAlbedoMap(const glm::vec2& uv) const
		{
			float4 color = ReadTexture(m_MaterialData.albedoMapId, uv);
			glm::vec3 albedo(color.x, color.y, color.z);
			return Spectrum(albedo);
		}
		
		INLINE GPU_ONLY glm::vec3 GetNormalMap(const glm::vec3& normal, const glm::vec2& uv) const
		{
			float4 nor = ReadTexture(m_MaterialData.normalMapId, uv);
			glm::vec3 tan, bitan;

			CoordinateSystem(normal, tan, bitan);

			return glm::normalize(nor.x * tan + nor.y * bitan + nor.z * normal);
		}
	public:
		MaterialData m_MaterialData;
		BSDF m_BSDF;
	};
}

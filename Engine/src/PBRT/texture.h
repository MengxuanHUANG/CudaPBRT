#pragma once

#include "pbrtDefine.h"
#include "pbrtUtilities.h"
#include "spectrum.h"

namespace CudaPBRT
{
	class CudaTexture
	{
	public:
		CPU_ONLY CudaTexture(const char* path, bool flip_v = false);
		CPU_ONLY ~CudaTexture();

		INLINE CPU_ONLY const float* GetRawData() const { return raw_data; }
		INLINE CPU_ONLY CudaTexObj GetTextureObject() const { return m_TexObj; }
		INLINE CPU_ONLY std::pair<int, int> GetDim() const { return { m_Width, m_Height }; }

	protected:
		CPU_ONLY void InitTexture(const char* path, bool flip_v = false);
		CPU_ONLY void FreeTexture();

	protected:
		int m_Width, m_Height;
		float* raw_data;
		CudaTexObj m_TexObj;
		CudaArray m_CudaArray;
	public:
		static uPtr<CudaTexture> CreateCudaTexture(const char* path, bool flip_v = false);
	};

	class GPUTexture
	{
	public:
		CPU_GPU GPUTexture() {}

		CPU_GPU GPUTexture(CudaTexObj tex_obj);
		INLINE GPU_ONLY CudaTexObj GetTexObj() const { return m_TexObj; }
		INLINE CPU_ONLY void SetTexObj(CudaTexObj texObj) { m_TexObj = texObj; }
		INLINE GPU_ONLY float4 GetColor(const glm::vec2 uv) const
		{
			return ReadTexture(m_TexObj, uv);
		}

	public:
		CudaTexObj m_TexObj;
	};

	class EnvironmentMap : public GPUTexture
	{
	public:
		CPU_GPU EnvironmentMap() {}

		CPU_GPU EnvironmentMap(CudaTexObj tex_obj);

		INLINE GPU_ONLY float4 GetIrradiance(const glm::vec3 wiW) const
		{
			return ReadTexture(m_TexObj, GetUVFromWiW(wiW));
		}

		INLINE GPU_ONLY static glm::vec2 GetUVFromWiW(const glm::vec3 wiW)
		{
			return glm::vec2(glm::atan(wiW.z, wiW.x), glm::asin(wiW.y)) * glm::vec2(Inv2Pi, InvPi) + glm::vec2(0.5f);
		}

		INLINE CPU_GPU static glm::vec3 GetWiWFromUV(const glm::vec2& uv) 
		{
			glm::vec2 temp = (uv - glm::vec2(0.5f)) * glm::vec2(2.f * Pi, Pi);

			float y = glm::cos(temp.y);
			float x = glm::sin(temp.y) * glm::cos(temp.x);
			float z = glm::sin(temp.y) * glm::sin(temp.x);
			
			return { x, y, z };
		}
	};
}
#pragma once

#include "pbrtDefine.h"
#include "spectrum.h"

namespace CudaPBRT
{
	class CudaTexture
	{
	public:
		CPU_ONLY CudaTexture(const char* path, bool flip_v = false);
		CPU_ONLY ~CudaTexture();

		INLINE CPU_ONLY CudaTexObj GetTextureObject() const { return m_TexObj; }
		INLINE CPU_ONLY std::pair<int, int> GetDim() const { return { m_Width, m_Height }; }

	protected:
		CPU_ONLY void InitTexture(const char* path, bool flip_v = false);
		CPU_ONLY void FreeTexture();

	protected:
		int m_Width, m_Height;
		CudaTexObj m_TexObj;
		CudaArray m_CudaArray;
	public:
		static uPtr<CudaTexture> CreateCudaTexture(const char* path, bool flip_v = false);
	};

	GPU_ONLY float4 ReadTexture(const CudaTexObj& tex_obj, const glm::vec2& uv);
	
	class GPUTexture
	{
	public:
		CPU_GPU GPUTexture(CudaTexObj tex_obj);
		INLINE GPU_ONLY CudaTexObj GetTexObj() const { return m_TexObj; }
		INLINE CPU_ONLY void SetTexObj(CudaTexObj texObj) { m_TexObj = texObj; }
		INLINE GPU_ONLY float4 GetColor(const glm::vec2 uv) const
		{
			return ReadTexture(m_TexObj, uv);
		}

	protected:
		CudaTexObj m_TexObj;
	};

	class EnvironmentMap : public GPUTexture
	{
	public:
		CPU_GPU EnvironmentMap(CudaTexObj tex_obj);

		INLINE GPU_ONLY float4 GetIrradiance(const glm::vec3 wiW) const
		{
			return ReadTexture(m_TexObj, GetUVFromWiW(wiW));
		}

		INLINE GPU_ONLY glm::vec2 GetUVFromWiW(const glm::vec3 wiW) const
		{
			return glm::vec2(glm::atan(wiW.z, wiW.x), glm::asin(wiW.y)) * glm::vec2(0.1591, 0.3183) + 0.5f;
		}
	};
}
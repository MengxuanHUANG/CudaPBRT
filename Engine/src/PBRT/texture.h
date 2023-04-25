#pragma once

#include "pbrtDefine.h"

namespace CudaPBRT
{
	class CudaTexture
	{
	public:
		CPU_ONLY CudaTexture(const char* path);
		CPU_ONLY ~CudaTexture();

		INLINE CPU_ONLY CudaTexObj GetTextureObject() const { return m_TexObj; }
		INLINE CPU_ONLY std::pair<int, int> GetDim() const { return { m_Width, m_Height }; }

	protected:
		CPU_ONLY void InitTexture(const char* path);
		CPU_ONLY void FreeTexture();

	protected:
		int m_Width, m_Height;
		CudaTexObj m_TexObj;
		CudaArray m_CudaArray;
	public:
		static uPtr<CudaTexture> CreateCudaTexture(const char* path);
	};
}
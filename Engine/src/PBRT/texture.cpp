#include "texture.h"

#include <stb_image.h>

namespace CudaPBRT
{
	CudaTexture::CudaTexture(const char* path)
		:m_TexObj(0), m_CudaArray()
	{
		InitTexture(path);
	}

	CudaTexture::~CudaTexture()
	{
		FreeTexture();
	}

	void CudaTexture::InitTexture(const char* path)
	{
        // free original texture if necessary
        if (m_TexObj != 0)
        {
            FreeTexture();
        }
        
        // read image from file       
       
        float* image_data = stbi_loadf(path, &m_Width, &m_Height, NULL, 4);
        ASSERT(image_data);

        // Allocate CUDA array in device memory
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
        cudaMallocArray(&m_CudaArray, &channelDesc, m_Width, m_Height);

        // Set pitch of the source (the width in memory in bytes of the 2D array pointed
        // to by src, including padding), we dont have any padding
        const size_t spitch = m_Width * sizeof(float4);
        // Copy data located at address h_data in host memory to device memory
        cudaMemcpy2DToArray(m_CudaArray, 0, 0, image_data, spitch, m_Width * sizeof(float4), m_Height, cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = m_CudaArray;

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;

        // Create texture object
        cudaCreateTextureObject(&m_TexObj, &resDesc, &texDesc, NULL);

        // free image array
        stbi_image_free(image_data);
	}

	void CudaTexture::FreeTexture()
	{
		// Destroy texture object
		cudaDestroyTextureObject(m_TexObj);

		// Free device memory
		cudaFreeArray(m_CudaArray);
	}

    uPtr <CudaTexture> CudaTexture::CreateCudaTexture(const char* path)
	{
        return mkU<CudaTexture>(path);
	}
}
#pragma once

#include "Camera/Camera.h"

#include <limits>
#include <cuda_runtime.h>

#ifndef MachineEpsilon
#define MachineEpsilon 0.5f * std::numeric_limits<float>::epsilon()
#endif // !MachineEpsilon

namespace CudaPBRT
{
	inline float gamma(int n)
	{
		return n * MachineEpsilon / (1.f - n * MachineEpsilon);
	}

	class CudaPathTracer
	{
	public:
		CudaPathTracer();
		virtual ~CudaPathTracer();

		virtual void InitCuda(PerspectiveCamera& camera, int device = 0);
		virtual void FreeCuda();
		virtual void Run();

		virtual void UpdateCamera(PerspectiveCamera& camera);
		virtual unsigned int GetDisplayTextureId() const { return m_DisplayImage; }

	protected:
		int m_CurrentId; // indicate the idx of current frame in device_image
		int m_Iteration; // number of iteration

		int width, height;

		// device handler
		unsigned int m_DisplayImage = 0;

		PerspectiveCamera* device_camera;
		uchar4* device_image;
		uchar4* host_image;
	};
}

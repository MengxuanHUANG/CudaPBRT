#pragma once

#include "pbrtDefine.h"
#include "Camera/Camera.h"

#include <vector>

namespace CudaPBRT
{
	class Shape;
	class Material;
	class Light;

	template<typename T, typename DataType>
	void CreateArrayOnCude(T**& dev_array, size_t*& dev_count, std::vector<DataType>& host_data);

	template<typename T>
	void FreeArrayOnCuda(T**& device_array, size_t*& count);

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

		inline void ResetPRBT() { m_Iteration = 1; }

	public:
		int m_Iteration; // number of iteration

		int width, height;

		// texture handler
		unsigned int m_DisplayImage = 0;

		// device handler
		int* device_iteration;

		PerspectiveCamera* device_camera;
		uchar4* device_image;
		float3* device_hdr_image;

		uchar4* host_image;
		Shape** device_shapes = nullptr;
		Material** device_materials = nullptr;
		Light** device_lights = nullptr;

		size_t* device_shape_count = nullptr;
		size_t* device_material_count = nullptr;
		size_t* device_light_count = nullptr;
	};
}

#pragma once

#include "pbrtDefine.h"
#include "Camera/Camera.h"

#include <vector>

namespace CudaPBRT
{
	class Shape;
	class Material;
	
	struct ShapeData;
	struct MaterialData;

	class CudaPathTracer
	{
	public:
		CudaPathTracer();
		virtual ~CudaPathTracer();

		virtual void InitCuda(PerspectiveCamera& camera, int device = 0);
		virtual void CreateShapesOnCuda(std::vector<ShapeData>& shapeData);
		virtual void FreeShapesOnCuda();

		virtual void CreateMaterialsOnCuda(std::vector<MaterialData>& materialData);
		virtual void FreeMaterialsOnCuda();

		virtual void FreeCuda();
		virtual void Run();

		virtual void UpdateCamera(PerspectiveCamera& camera);
		virtual unsigned int GetDisplayTextureId() const { return m_DisplayImage; }

	protected:
		int m_CurrentId; // indicate the idx of current frame in device_image
		int m_Iteration; // number of iteration

		int width, height;

		// texture handler
		unsigned int m_DisplayImage = 0;

		// device handler
		PerspectiveCamera* device_camera;
		uchar4* device_image;
		uchar4* host_image;
		Shape** device_shapes;
		Material** device_materials;

		unsigned int* device_shape_count;
		unsigned int* device_material_count;
	};
}

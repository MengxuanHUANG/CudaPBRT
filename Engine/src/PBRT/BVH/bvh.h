#pragma once

#include "PBRT/pbrtDefine.h"
#include "PBRT/Shape/shape.h"

#include "boundingBox.h"
#include <vector>

namespace CudaPBRT
{
	struct BVHNode
	{
		int primitiveId = -1;
		int boundingBoxId = -1;
		int splitAxis = -1;
		int next = -1;

		CPU_GPU BVHNode()
		{
		}

		CPU_GPU BVHNode(int primitive, int bb, int axis = -1, int next = -1)
			: primitiveId(primitive), boundingBoxId(bb), splitAxis(axis), next(next)
		{}
	};

	void CreateBVH(std::vector<ShapeData>& shapeData, 
				   const std::vector<glm::vec3>& vertices,
				   std::vector<BoundingBox>& bounding_boxes, 
				   std::vector<BVHNode>& BVH);
}
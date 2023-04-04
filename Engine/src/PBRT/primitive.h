#pragma once

#include "PBRT/pbrtDefine.h"

namespace CudaPBRT
{
	class Shape;

	struct Primitive
	{
		Shape* shape;
		int material_id;
	};
}
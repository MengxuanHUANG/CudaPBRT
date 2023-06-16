#include "shape.h"

#include "Core/Utilities.h"

namespace CudaPBRT
{
	using namespace StringUtility;

	ShapeType Str2ShapeType(const char* str)
	{
		switch (StringUtility::hash_djb2a(str))
		{
		case "Sphere"_sh:
			return ShapeType::Sphere;
		case "Cube"_sh:
			return ShapeType::Cube;
		case "Triangle"_sh:
			return ShapeType::Triangle;
		default:
			return ShapeType::None;
		}
	}
}
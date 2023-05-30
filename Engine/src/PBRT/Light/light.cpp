#include "light.h"

#include "Core/Utilities.h"

namespace CudaPBRT
{
	using namespace StringUtility;

	LightType Str2LightType(const char* str)
	{
		switch (StringUtility::hash_djb2a(str))
		{
		case "ShapeLight"_sh:
			return LightType::ShapeLight;
		default:
			return LightType::None;
		}
	}
}
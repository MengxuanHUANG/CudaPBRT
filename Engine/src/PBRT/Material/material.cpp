#include "material.h"

#include "Core/Utilities.h"

namespace CudaPBRT
{
	using namespace StringUtility;

	MaterialType Str2MaterialType(const char* str)
	{
		switch (StringUtility::hash_djb2a(str))
		{
		case "LambertianReflection"_sh:
			return MaterialType::LambertianReflection;
		case "SpecularReflection"_sh:
			return MaterialType::SpecularReflection;
		case "SpecularTransmission"_sh:
			return MaterialType::SpecularTransmission;
		case "Glass"_sh:
			return MaterialType::Glass;
		case "MicrofacetReflection"_sh:
			return MaterialType::MicrofacetReflection;
		case "MetallicWorkflow"_sh:
			return MaterialType::MetallicWorkflow;
		default:
			return MaterialType::None;
		}
	}
}
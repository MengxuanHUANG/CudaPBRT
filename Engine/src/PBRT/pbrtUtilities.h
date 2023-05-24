#pragma once

#include "Core/Core.h"
#include "pbrtDefine.h"
#include <glm/glm.hpp>

namespace CudaPBRT
{
	INLINE CPU_GPU float AbsDot(const glm::vec3& wi, const glm::vec3& nor) { return glm::abs(glm::dot(wi, nor)); }

	INLINE CPU_GPU float CosTheta(const glm::vec3& w) { return w.z; }
	INLINE CPU_GPU float Cos2Theta(const glm::vec3& w) { return w.z * w.z; }
	INLINE CPU_GPU float Sin2Theta(const glm::vec3& w) { return glm::max(0.f, 1.f - Cos2Theta(w)); }
	INLINE CPU_GPU float SinTheta(const glm::vec3& w) { return glm::sqrt(Sin2Theta(w)); }
	INLINE CPU_GPU float TanTheta(const glm::vec3& w) { return SinTheta(w) / CosTheta(w); }
	INLINE CPU_GPU float Tan2Theta(const glm::vec3& w) { return Sin2Theta(w) / Cos2Theta(w); }

	INLINE CPU_GPU float CosPhi(const glm::vec3& w) {
		float sinTheta = SinTheta(w);
		return (sinTheta == 0) ? 1 : glm::clamp(w.x / sinTheta, -1.f, 1.f);
	}
	INLINE CPU_GPU float SinPhi(const glm::vec3& w) {
		float sinTheta = SinTheta(w);
		return (sinTheta == 0) ? 0 : glm::clamp(w.y / sinTheta, -1.f, 1.f);
	}

	INLINE CPU_GPU float Cos2Phi(const glm::vec3& w) { return CosPhi(w) * CosPhi(w); }
	INLINE CPU_GPU float Sin2Phi(const glm::vec3& w) { return SinPhi(w) * SinPhi(w); }

	INLINE CPU_GPU float AbsCosTheta(const glm::vec3& w) { return glm::abs(w.z); }
	INLINE CPU_GPU  float PowerHeuristic(int nf, float fPdf, int ng, float gPdf)
	{
		float f = static_cast<float>(nf) * fPdf;
		float g = static_cast<float>(ng) * gPdf;

		return (f * f) / (f * f + g * g);
	}

	INLINE CPU_GPU void CoordinateSystem(const glm::vec3& normal, glm::vec3& tangent, glm::vec3& bitangent)
	{
		glm::vec3 up = glm::abs(normal.z) < 0.999f ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(1.f, 0.f, 0.f);
		tangent = glm::normalize(glm::cross(up, normal));
		bitangent = glm::cross(normal, tangent);
	}

	INLINE CPU_GPU glm::mat3 LocalToWorld(const::glm::vec3& nor)
	{
		glm::vec3 tan, bit;
		CoordinateSystem(nor, tan, bit);
		return glm::mat3(tan, bit, nor);
	}
	INLINE CPU_GPU glm::mat3 WorldToLocal(const::glm::vec3& nor)
	{
		return glm::transpose(LocalToWorld(nor));
	}

	template<typename T>
	INLINE CPU_GPU T BarycentricInterpolation(const T& c1, const T& c2, const T& c3, float u, float v)
	{
		return u * c1 + v * c2 + (1.f - u - v) * c3;
	}

	GPU_ONLY float4 ReadTexture(const CudaTexObj& tex_obj, const glm::vec2& uv);
}
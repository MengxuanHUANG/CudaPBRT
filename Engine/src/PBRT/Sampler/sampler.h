#pragma once
#include "PBRT/pbrtDefine.h"

namespace CudaPBRT
{
    CPU_GPU void coordinateSystem(const glm::vec3& v1, glm::vec3& v2, glm::vec3& v3) {
        if (glm::abs(v1.x) > glm::abs(v1.y))
            v2 = glm::vec3(-v1.z, 0, v1.x) / glm::sqrt(v1.x * v1.x + v1.z * v1.z);
        else
            v2 = glm::vec3(0, v1.z, -v1.y) / glm::sqrt(v1.y * v1.y + v1.z * v1.z);
        v3 = glm::cross(v1, v2);
    }

    CPU_GPU glm::mat3 LocalToWorld(const::glm::vec3& nor) 
    {
        glm::vec3 tan, bit;
        coordinateSystem(nor, tan, bit);
        return glm::mat3(tan, bit, nor);
    }
    CPU_GPU glm::mat3 WorldToLocal(const::glm::vec3& nor)
    {
        return glm::transpose(LocalToWorld(nor));
    }

	class Sampler
	{
	public:
		CPU_GPU static glm::vec3 SquareToDiskConcentric(const glm::vec2& xi)
		{
            glm::vec2 offset = xi * 2.f - 1.f;

            if (offset.x != 0.f || offset.y != 0.f)
            {
                float theta, r;
                if (glm::abs(offset.x) > glm::abs(offset.y))
                {
                    r = offset.x;
                    theta = CudaPBRT::PiOver4 * (offset.y / offset.x);
                }
                else
                {
                    r = offset.y;
                    theta = CudaPBRT::PiOver2 - CudaPBRT::PiOver4 * (offset.x / offset.y);
                }
                return r * glm::vec3(glm::cos(theta), glm::sin(theta), 0);
            }
            return glm::vec3(0.);
		}

		INLINE CPU_GPU static glm::vec3 SquareToHemisphereCosine(const glm::vec2& xi)
		{
			glm::vec3 result = SquareToDiskConcentric(xi);
			result.z = glm::sqrt(glm::max(0.f, 1.f - result.x * result.x - result.y * result.y));
            result.z = glm::max(result.z, 0.01f);

			return result;
		 }

		INLINE CPU_GPU static float SquareToHemisphereCosinePDF(const glm::vec3& sample)
		{
			return sample.z * CudaPBRT::InvPi; // cos(theta) / PI
		}
	};
	
}
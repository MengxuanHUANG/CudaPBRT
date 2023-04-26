#pragma once
#include "PBRT/pbrtDefine.h"

namespace CudaPBRT
{
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
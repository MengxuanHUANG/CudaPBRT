#pragma once

#include "shape.h"

namespace CudaPBRT
{
    /**
    * Default position: (0, 0, 0)
    * Default radius: 1
    */
	class Sphere: public Shape
	{
	public:
		GPU_ONLY Sphere(const ShapeData& data)
			: Shape(data)
		{}

		GPU_ONLY virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const override
		{
            glm::vec3 local_origin = glm::vec3(m_TransformInv * glm::vec4(ray.O, 1.0f));
            glm::vec3 local_dir    = glm::vec3(m_TransformInv * glm::vec4(ray.DIR, 0.0f));

            Ray localRay(local_origin, local_dir);

            // calculate intersection for sphere and the ray
            float A = glm::dot(localRay.DIR, localRay.DIR);
            float B = 2.f * glm::dot(localRay.DIR, localRay.O);
            float C = glm::dot(localRay.O, localRay.O) - 1.f;

            float delta = B * B - 4.f * A * C;
            
            if (delta >= 0)
            {
                float t1 = 0.5f * (-B + glm::sqrt(delta)) / A;
                float t2 = 0.5f * (-B - glm::sqrt(delta)) / A;
                float t = t2 > 0 ? t2 : t1;

                glm::vec3 local_pos = localRay * t; // also the local normal
                glm::vec3 normal = m_TransformInvTranspose * local_pos;
                normal = glm::normalize(normal);

                intersection = Intersection(t, ray * t, normal);

                return true;
            }

            return false;
		}
	};
}
#pragma once

#include "shape.h"

namespace CudaPBRT
{
    /**
    * Default position: (0, 0, 0)
    * Default Surface normal: (0, 0, 1)
    * Default size [width, height] [1, 1]
    */
    class Square : public Shape
    {
    public:
        CPU_GPU Square(const ShapeData& data)
            : Shape(data)
        {}

        CPU_GPU virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const override
        {
            glm::vec3 local_origin = glm::vec3(m_TransformInv * glm::vec4(ray.O, 1.0f));
            glm::vec3 local_dir = glm::vec3(m_TransformInv * glm::vec4(ray.DIR, 0.0f));

            Ray localRay(local_origin, local_dir);

            float t = -localRay.O.z / localRay.DIR.z; // t = -o.z / dir.z

            if (t > 0.f) // ray is toward the square
            {
                glm::vec3 local_p = localRay * t;

                if (local_p.x >= -0.5f &&
                    local_p.x <=  0.5f &&
                    local_p.y >= -0.5f &&
                    local_p.y <=  0.5f)
                {
                    glm::vec3 normal = m_TransformInvTranspose * glm::vec3(0.f, 0.f, 1.f);
                    intersection = Intersection(t, ray * t, glm::normalize(normal));
                    return true;
                }
            }

            return false;
        }
    };
}
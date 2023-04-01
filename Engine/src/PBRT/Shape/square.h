#pragma once

#include "shape.h"

namespace CudaPBRT
{
    class Square : public Shape
    {
    public:
        GPU_ONLY Square(const ShapeData& data)
            : Shape(data)
        {}

        GPU_ONLY virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const override
        {
            glm::vec3 local_origin = glm::vec3(m_TransformInv * glm::vec4(ray.O, 1.0f));
            glm::vec3 local_dir = glm::vec3(m_TransformInv * glm::vec4(ray.DIR, 0.0f));

            Ray localRay(local_origin, local_dir);



            return false;
        }
    };
}
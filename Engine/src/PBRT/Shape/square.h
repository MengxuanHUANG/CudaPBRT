#pragma once

#include "shape.h"

namespace CudaPBRT
{
    /**
    * Default position: (0, 0, 0)
    * Default Surface normal: (0, 0, 1)
    * Default size [width, height] [2, 2]
    */
    class Square : public Shape
    {
    public:
        CPU_GPU Square(const ShapeData& data)
            : Shape(data), shapeData(data)
        {
            Shape::ComputeTransforms(data.translation, data.rotation, data.scale, m_Transform, m_TransformInv, m_TransformInvTranspose);
        }

        CPU_GPU virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const override
        {
            glm::vec3 local_origin = glm::vec3(m_TransformInv * glm::vec4(ray.O, 1.0f));
            glm::vec3 local_dir = glm::vec3(m_TransformInv * glm::vec4(ray.DIR, 0.0f));

            Ray localRay(local_origin, local_dir);

            float t = -localRay.O.z / localRay.DIR.z; // t = -o.z / dir.z

            if (localRay.DIR.z < 0.f && t > 0.f) // ray is toward the square
            {
                glm::vec3 local_p = localRay * t;

                if (local_p.x >= -0.5f &&
                    local_p.x <=  0.5f &&
                    local_p.y >= -0.5f &&
                    local_p.y <=  0.5f)
                {
                    intersection = Intersection(t, ray * t, ComputeNormal());
                    return true;
                }
            }

            return false;
        }
        
        CPU_GPU virtual glm::vec3 GetNormal(const glm::vec3& p) const override
        {
            return ComputeNormal();
        }

        CPU_GPU virtual float Area() const 
        {
            return glm::abs(shapeData.scale.x * shapeData.scale.y);
        }

        CPU_GPU virtual glm::vec3 Sample(const glm::vec2& xi) const override
        {
            glm::vec2 local_p = (xi - 0.5f); // map point to [-1, 1]
            return glm::vec3(m_Transform * glm::vec4(local_p, 0.f, 1.f)); // transform from local to world
        }

        INLINE CPU_ONLY static BoundingBox GetWorldBounding(const ShapeData& data)
        {
            glm::vec3 v[4]{glm::vec3(1, 1, 0), glm::vec3(1, -1, 0), glm::vec3(-1, 1, 0), glm::vec3(-1, -1, 0)};
            
            glm::mat4 transform;

            Shape::ComputeTransform(data.translation, data.rotation, data.scale, transform);

            for (int i = 0; i < 4; ++i)
            {
                v[i] = glm::vec3(transform * glm::vec4(v[i], 1.f));
            }

            return {glm::min(glm::min(glm::min(v[0], v[1]), v[2]), v[3]), glm::max(glm::max(glm::max(v[0], v[1]), v[2]), v[3]) };
        }

    protected:
        INLINE CPU_GPU glm::vec3 ComputeNormal() const
        {
            return glm::normalize(m_TransformInvTranspose * glm::vec3(0.f, 0.f, 1.f));
        }

    protected:
        glm::mat4 m_Transform;
        glm::mat4 m_TransformInv;
        glm::mat3 m_TransformInvTranspose;

        ShapeData shapeData;
    };
}
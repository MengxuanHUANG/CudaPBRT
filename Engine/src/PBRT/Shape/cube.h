#pragma once

#include "shape.h"

namespace CudaPBRT
{
    /**
    * Default position: (0, 0, 0)
    * Default length: 1
    */
    class Cube : public Shape
    {
    public:
        CPU_GPU Cube(const ShapeData& data)
            : Shape(data)
        {}

        CPU_GPU virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const override
        {
            glm::vec3 local_origin = glm::vec3(m_TransformInv * glm::vec4(ray.O, 1.0f));
            glm::vec3 local_dir = glm::vec3(m_TransformInv * glm::vec4(ray.DIR, 0.0f));

            Ray localRay(local_origin, local_dir);

            glm::vec3 pMin(-0.5f);
            glm::vec3 pMax(0.5f);

            glm::vec3 invDir = 1.f / localRay.DIR;
            glm::vec3 tNear = (pMin - localRay.O) * invDir;
            glm::vec3 tFar  = (pMax - localRay.O) * invDir;

            glm::vec3 tmin = glm::min(tNear, tFar);
            glm::vec3 tmax = glm::max(tNear, tFar);

            float t0 = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
            float t1 = glm::min(glm::min(tmax.x, tmax.y), tmax.z);

            if (t0 > t1) return false;
            
            if (t0 > 0.f)
            {
                glm::vec3 v1(tmin.y, tmin.z, tmin.x);
                glm::vec3 v2(tmin.z, tmin.x, tmin.y);

                glm::vec3 normal = -glm::sign(localRay.DIR) * glm::step(v1, tmin) * glm::step(v2, tmin);
                normal = glm::normalize(m_TransformInvTranspose * normal);

                intersection = Intersection(t0, ray * t0, normal);
                return true;
            }
            if (t1 > 0.f)
            {
                glm::vec3 v1(tmax.y, tmax.z, tmax.x);
                glm::vec3 v2(tmax.z, tmax.x, tmax.y);

                glm::vec3 normal = -glm::sign(localRay.DIR) * glm::step(tmax, v1) * glm::step(tmax, v2);
                normal = glm::normalize(m_TransformInvTranspose * normal);
                intersection = Intersection(t1, ray * t1, normal);
                return true;
            }

            return false;
        }
    };
}
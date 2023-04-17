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
            : Shape(data), shapeData(data)
        {
            Shape::ComputeTransforms(data.translation, data.rotation, data.scale, m_Transform, m_TransformInv, m_TransformInvTranspose);
        }

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
    
        CPU_GPU virtual glm::vec3 GetNormal(const glm::vec3& p) const
        {
            return ComputeNormal(p);
        }

        INLINE CPU_ONLY static BoundingBox GetWorldBounding(const ShapeData& data)
        {
            glm::vec3 v[8]{ glm::vec3( 0.5,  0.5,  0.5), 
                            glm::vec3( 0.5, -0.5,  0.5), 
                            glm::vec3(-0.5,  0.5,  0.5), 
                            glm::vec3(-0.5, -0.5,  0.5),
                            glm::vec3( 0.5,  0.5, -0.5),
                            glm::vec3( 0.5, -0.5, -0.5),
                            glm::vec3(-0.5,  0.5, -0.5),
                            glm::vec3(-0.5, -0.5, -0.5) 
            };

            glm::mat4 transform;

            Shape::ComputeTransform(data.translation, data.rotation, data.scale, transform);
            
            glm::vec3 p_min(FloatMax);
            glm::vec3 p_max(FloatMin);

            for (int i = 0; i < 8; ++i)
            {
                v[i] = glm::vec3(transform * glm::vec4(v[i], 1.f));
                p_min = glm::min(p_min, v[i]);
                p_max = glm::max(p_max, v[i]);
            }

            return { p_min, p_max };
        }

    protected:
        INLINE CPU_GPU glm::vec3 ComputeNormal(const glm::vec3& p) const
        {
            glm::vec3 local_p = glm::vec3(m_TransformInv * glm::vec4(p, 1.0f));
            glm::vec3 normal(0.f);
            int max_axis = 0;
            if (glm::abs(local_p[max_axis]) < glm::abs(local_p[1])) max_axis = 1;
            if (glm::abs(local_p[max_axis]) < glm::abs(local_p[2])) max_axis = 2;
            normal[max_axis] = glm::sign(local_p[max_axis]);

            return normal;
        }
    
    protected:
        glm::mat4 m_Transform;
        glm::mat4 m_TransformInv;
        glm::mat3 m_TransformInvTranspose;

        ShapeData shapeData;
    };
}
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
        CPU_GPU Sphere(const ShapeData& data)
			: Shape(data), shapeData(data)
		{
            Shape::ComputeTransforms(data.translation, data.rotation, data.scale, m_Transform, m_TransformInv, m_TransformInvTranspose);
        }

        CPU_GPU virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const override
		{
            glm::vec3 local_origin = glm::vec3(m_TransformInv * glm::vec4(ray.O, 1.0f));
            glm::vec3 local_dir    = glm::vec3(m_TransformInv * glm::vec4(ray.DIR, 0.0f));

            Ray localRay(local_origin, local_dir);

            // calculate intersection for sphere and the ray
            float A = glm::dot(localRay.DIR, localRay.DIR);
            float B = 2.f * glm::dot(localRay.DIR, localRay.O);
            float C = glm::dot(localRay.O, localRay.O) - 1.f;
            float t0 = 0.f, t1 = 0.f;

            SolveQuadratic(A, B, C, t0, t1);

            float t = (t0 > 0.f ? t0 : t1 > 0.f ? t1 : -1.f);

            if (t > 0.f)
            {
                glm::vec3 normal = glm::normalize(m_TransformInvTranspose * (localRay * t));

                intersection = Intersection(t, ray * t, normal);
                return true;
            }

            return false;
		}

        CPU_GPU virtual float SimpleIntersection(const Ray& ray) const override
        {
            glm::vec3 local_origin = glm::vec3(m_TransformInv * glm::vec4(ray.O, 1.0f));
            glm::vec3 local_dir = glm::vec3(m_TransformInv * glm::vec4(ray.DIR, 0.0f));

            Ray localRay(local_origin, local_dir);

            // calculate intersection for sphere and the ray
            float A = glm::dot(localRay.DIR, localRay.DIR);
            float B = 2.f * glm::dot(localRay.DIR, localRay.O);
            float C = glm::dot(localRay.O, localRay.O) - 1.f;
            float t0 = 0.f, t1 = 0.f;

            SolveQuadratic(A, B, C, t0, t1);

            return (t0 > 0.f ? t0 : (t1 > 0.f ? t1 : -1.f));
        }

        CPU_GPU virtual glm::vec3 GetNormal(const glm::vec3& p) const override
        {
            return ComputeNormal(p);
        }

        INLINE CPU_ONLY static BoundingBox GetWorldBounding(const ShapeData& data)
        {
            BoundingBox box;
            
            glm::mat4 transform;

            Shape::ComputeTransform(data.translation, data.rotation, data.scale, transform);
            transform = glm::transpose(transform);

            float cx = transform[0][3];
            float dx = glm::sqrt(glm::dot(transform[0], transform[0]));
            box.m_Min.x = cx - dx;
            box.m_Max.x = cx + dx;

            float cy = transform[1][3];
            float dy = glm::sqrt(glm::dot(transform[1], transform[1]));
            box.m_Min.y = cy - dy;
            box.m_Max.y = cy + dy;

            float cz = transform[2][3];
            float dz = glm::sqrt(glm::dot(transform[2], transform[2]));
            box.m_Min.z = cz - dz;
            box.m_Max.z = cz + dz;

            return box;
        }

    protected:
        INLINE CPU_GPU static void SolveQuadratic(float A, float B, float C, float& t0, float& t1) 
        {
            float invA = 1.f / A;
            B *= invA;
            C *= invA;
            float neg_halfB = -B * 0.5f;
            float u2 = neg_halfB * neg_halfB - C;
            float u = u2 < 0.f ? (neg_halfB = 0.f) : glm::sqrt(u2);
            t0 = neg_halfB - u;
            t1 = neg_halfB + u;
        }
        INLINE CPU_GPU glm::vec3 ComputeNormal(const glm::vec3& p) const
        {
            return glm::normalize(p - shapeData.translation);
        }

    protected:
        glm::mat4 m_Transform;
        glm::mat4 m_TransformInv;
        glm::mat3 m_TransformInvTranspose;

        ShapeData shapeData;
	};

}
#pragma once

#include "shape.h"

namespace CudaPBRT
{
    class Triangle : public Shape
    {
    public:
        CPU_GPU Triangle(const ShapeData& data)
            : Shape(data), 
              m_Vertices(data.vertices),
              m_VerticesId(data.verticeId)
        {}

        CPU_GPU virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const override
        {
            // Moller¨CTrumbore intersection
            glm::vec2 uv;

            const glm::vec3& v0 = m_Vertices[m_VerticesId[0]];
            
            glm::vec3 edge01 = m_Vertices[m_VerticesId[1]] - v0;
            glm::vec3 edge02 = m_Vertices[m_VerticesId[2]] - v0;
            glm::vec3 pvec = glm::cross(ray.DIR, edge02);

            float det = glm::dot(pvec, edge01);

            if (glm::abs(det) < CudaPBRT::FloatEpsilon) // ray is perpendicular to the plane contain the triangle
            {
                return false;
            }
            
            glm::vec3 tvec = ray.O - v0;
            if (det < 0.f)
            {
                det = -det;
                tvec = -tvec;
            }

            uv.x = glm::dot(tvec, pvec);
            if (uv.x < 0.f || uv.x > det)
            {
                return false;
            }

            glm::vec3 qvec = glm::cross(tvec, edge01);

            uv.y = glm::dot(ray.DIR, qvec);
            if (uv.y < 0.f || uv.x + uv.y > det) 
            {
                return false;
            }
            float inv_det = 1.f / det;
            uv *= inv_det;
            float t = glm::dot(edge02, qvec) * inv_det;

            if (t > 0.f)
            {
                intersection = Intersection(t, ray * t, glm::normalize(glm::cross(edge01, edge02)));
                return true;
            }
            else
            {
                return false;
            }
        }
        
        CPU_GPU virtual glm::vec3 GetNormal(const glm::vec3& p) const override
        {
            const glm::vec3& v0 = m_Vertices[m_VerticesId[0]];
            const glm::vec3& v1 = m_Vertices[m_VerticesId[1]];
            const glm::vec3& v2 = m_Vertices[m_VerticesId[2]];

            return glm::normalize(glm::cross(v1 - v0, v2 - v0));
        }
    protected:
        glm::vec3* m_Vertices;
        glm::ivec3 m_VerticesId;
    };
}
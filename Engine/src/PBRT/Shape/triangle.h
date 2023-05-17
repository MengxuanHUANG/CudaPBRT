#pragma once

#include "shape.h"
#include "PBRT/pbrtUtilities.h"
#include <array>

namespace CudaPBRT
{
    class Triangle : public Shape
    {
    public:
        CPU_GPU Triangle(const ShapeData& data)
            : Shape(data), 
              m_Vertices(data.vertices), 
              m_Normals(data.normals),
              m_UVs(data.uvs),
              m_Triangle(data.triangle)
        {}

        CPU_GPU virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const override
        {
            // Moller¨CTrumbore intersection
            glm::vec2 uv;

            const glm::vec3& v0 = m_Vertices[m_Triangle.vId[0]];
            
            glm::vec3 edge01 = m_Vertices[m_Triangle.vId[1]] - v0;
            glm::vec3 edge02 = m_Vertices[m_Triangle.vId[2]] - v0;
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
                intersection.normal = m_Triangle.nId[0] >= 0 ? BarycentricInterpolation<glm::vec3>(m_Normals[m_Triangle.nId[1]],
                                                                                                   m_Normals[m_Triangle.nId[2]],
                                                                                                   m_Normals[m_Triangle.nId[0]], uv.x, uv.y):
                                                               glm::cross(edge01, edge02);
                intersection.normal = glm::normalize(intersection.normal);
                intersection.t = t;
                intersection.p = ray * t;

                intersection.uv = m_Triangle.uvId[0] >= 0 ? BarycentricInterpolation<glm::vec2>(m_UVs[m_Triangle.uvId[1]],
                                                                                                m_UVs[m_Triangle.uvId[2]],
                                                                                                m_UVs[m_Triangle.uvId[0]], uv.x, uv.y) :
                                                            uv;
                return true;
            }
            else
            {
                return false;
            }
        }
        
        CPU_GPU virtual glm::vec3 GetNormal(const glm::vec3& p) const override
        {
            const glm::vec3& v0 = m_Vertices[m_Triangle.vId[0]];
            const glm::vec3& v1 = m_Vertices[m_Triangle.vId[1]];
            const glm::vec3& v2 = m_Vertices[m_Triangle.vId[2]];

            return glm::normalize(glm::cross(v1 - v0, v2 - v0));
        }

        INLINE CPU_ONLY static BoundingBox GetWorldBounding(const std::array<glm::vec3, 3>& v)
        { 
            return { glm::min(glm::min(v[0], v[1]), v[2]), glm::max(glm::max(v[0], v[1]), v[2]) };
        }
    protected:
        glm::vec3* m_Vertices;
        glm::vec3* m_Normals;
        glm::vec2* m_UVs;

        TriangleData m_Triangle;
    };
}
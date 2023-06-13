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
        
        CPU_GPU virtual float SimpleIntersection(const Ray& ray) const override
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

            return glm::dot(edge02, qvec) * inv_det;
        }

        CPU_GPU virtual glm::vec3 GetNormal(const glm::vec3& p) const override
        {
            if (m_Triangle.nId[0] >= 0)
            {
                return glm::normalize(GetBarycentricInterpolation<glm::vec3>(m_Normals[m_Triangle.nId[0]], m_Normals[m_Triangle.nId[1]], m_Normals[m_Triangle.nId[2]], p));
            }
            else
            {
                const glm::vec3& v0 = m_Vertices[m_Triangle.vId[0]];

                glm::vec3 edge01 = m_Vertices[m_Triangle.vId[1]] - v0;
                glm::vec3 edge02 = m_Vertices[m_Triangle.vId[2]] - v0;

                return glm::normalize(glm::cross(edge01, edge02));
            }
        }
        
        CPU_GPU virtual glm::vec3 Sample(const glm::vec2& xi) const override
        {
            const glm::vec3& v0 = m_Vertices[m_Triangle.vId[0]];

            glm::vec3 edge01 = m_Vertices[m_Triangle.vId[1]] - v0;
            glm::vec3 edge02 = m_Vertices[m_Triangle.vId[2]] - v0;

            if (xi.x + xi.y <= 1.f)
            {
                return xi.x * edge02 + xi.y * edge01 + v0;
            }
            else
            {
                return (1.f - xi.x) * edge02 + (1.f - xi.y) * edge01 + v0;
            }
        }

        CPU_GPU virtual float Area() const override
        {
            const glm::vec3& v0 = m_Vertices[m_Triangle.vId[0]];
            const glm::vec3& v1 = m_Vertices[m_Triangle.vId[1]];
            const glm::vec3& v2 = m_Vertices[m_Triangle.vId[2]];

            return glm::length(glm::cross(v1 - v0, v2 - v0));
        }

        CPU_GPU virtual glm::vec2 GetUV(const glm::vec3& p) const override
        {
            return GetBarycentricInterpolation<glm::vec2>(m_UVs[m_Triangle.uvId[0]], m_UVs[m_Triangle.uvId[1]], m_UVs[m_Triangle.uvId[2]], p);
        }

        INLINE CPU_ONLY static BoundingBox GetWorldBounding(const std::array<glm::vec3, 3>& v)
        { 
            return { glm::min(glm::min(v[0], v[1]), v[2]), glm::max(glm::max(v[0], v[1]), v[2]) };
        }

    protected:
        template<typename T>
        INLINE CPU_GPU T GetBarycentricInterpolation(const T& x0, const T& x1, const T& x2, const glm::vec3& p) const
        {
            const glm::vec3& v0 = m_Vertices[m_Triangle.vId[0]];
            const glm::vec3& v1 = m_Vertices[m_Triangle.vId[1]];
            const glm::vec3& v2 = m_Vertices[m_Triangle.vId[2]];

            float a = Area();
            float a0 = glm::length(glm::cross(v1 - p, v2 - p)) / a;
            float a1 = glm::length(glm::cross(v2 - p, v0 - p)) / a;
            float a2 = glm::length(glm::cross(v0 - p, v1 - p)) / a;

            return a0 * x0 + a1 * x1 + a2 * x2;
        }
    protected:
        glm::vec3* m_Vertices;
        glm::vec3* m_Normals;
        glm::vec2* m_UVs;

        TriangleData m_Triangle;
    };
}
#pragma once

#include "shape.h"
#include "PBRT/pbrtUtilities.h"
#include <array>

namespace CudaPBRT
{
#define m_V m_UnionData.triangle_data.vertices
#define m_N m_UnionData.triangle_data.normals
#define m_UV m_UnionData.triangle_data.uvs

    class Triangle : public Shape
    {
    public:
        CPU_GPU Triangle(const ShapeData& data)
            : Shape(data)
        {
            m_V[0] = (data.vertices + data.triangle.vId[0]);
            m_V[1] = (data.vertices + data.triangle.vId[1]);
            m_V[2] = (data.vertices + data.triangle.vId[2]);

            m_N[0] = data.normals + data.triangle.nId[0];
            m_N[1] = data.normals + data.triangle.nId[1];
            m_N[2] = data.normals + data.triangle.nId[2];

            m_UV[0] = data.uvs + data.triangle.uvId[0];
            m_UV[1] = data.uvs + data.triangle.uvId[1];
            m_UV[2] = data.uvs + data.triangle.uvId[2];
        }

        CPU_GPU virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const override
        {
            // Moller¨CTrumbore intersection
            glm::vec2 uv;

            glm::vec3 edge01 = *m_V[1] - *m_V[0];
            glm::vec3 edge02 = *m_V[2] - *m_V[0];
            glm::vec3 pvec = glm::cross(ray.DIR, edge02);

            float det = glm::dot(pvec, edge01);

            if (glm::abs(det) < CudaPBRT::FloatEpsilon) // ray is perpendicular to the plane contain the triangle
            {
                return false;
            }
            
            glm::vec3 tvec = ray.O - *m_V[0];
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
                intersection.normal = glm::normalize(BarycentricInterpolation<glm::vec3>(*m_N[1], *m_N[2], *m_N[0], uv.x, uv.y));

                intersection.t = t;
                intersection.p = ray * t;

                intersection.uv = BarycentricInterpolation<glm::vec2>(*m_UV[1], *m_UV[2], *m_UV[0], uv.x, uv.y);

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

            glm::vec3 edge01 = *m_V[1] - *m_V[0];
            glm::vec3 edge02 = *m_V[2] - *m_V[0];
            glm::vec3 pvec = glm::cross(ray.DIR, edge02);

            float det = glm::dot(pvec, edge01);

            if (glm::abs(det) < CudaPBRT::FloatEpsilon) // ray is perpendicular to the plane contain the triangle
            {
                return false;
            }

            glm::vec3 tvec = ray.O - *m_V[0];
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
            return glm::normalize(GetBarycentricInterpolation<glm::vec3>(*m_N[0], *m_N[1], *m_N[2], p));
        }
        
        CPU_GPU virtual glm::vec3 Sample(glm::vec2 xi) const override
        {
            xi = (xi.x + xi.y > 1.f ? glm::vec2(1.f) - xi : xi);

            return xi.x * *m_V[2] + xi.y * *m_V[1] + (1.f - xi.x - xi.y) * *m_V[0];
        }

        CPU_GPU virtual float Area() const override
        {
            return 0.5f * glm::length(glm::cross(*m_V[1] - *m_V[0], *m_V[2] - *m_V[0]));
        }

        CPU_GPU virtual glm::vec2 GetUV(const glm::vec3& p) const override
        {
            return GetBarycentricInterpolation<glm::vec2>(*m_UV[0], *m_UV[1], *m_UV[0], p);
        }

        INLINE CPU_ONLY static BoundingBox GetWorldBounding(const std::array<glm::vec3, 3>& v)
        { 
            return { glm::min(glm::min(v[0], v[1]), v[2]), glm::max(glm::max(v[0], v[1]), v[2]) };
        }

    protected:
        template<typename T>
        INLINE CPU_GPU T GetBarycentricInterpolation(const T& x0, const T& x1, const T& x2, const glm::vec3& p) const
        {
            float a = Area();
            float a0 = glm::length(glm::cross(*m_V[1] - p, *m_V[2] - p)) / a;
            float a1 = glm::length(glm::cross(*m_V[2] - p, *m_V[0] - p)) / a;
            float a2 = glm::length(glm::cross(*m_V[0] - p, *m_V[1] - p)) / a;

            return a0 * x0 + a1 * x1 + a2 * x2;
        }
    };

#undef m_V
#undef m_N
#undef m_UV
}
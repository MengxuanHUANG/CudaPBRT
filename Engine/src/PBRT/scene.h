#pragma once

#include "pbrtDefine.h"
#include "intersection.h"
#include "Shape/shape.h"
#include "Material/material.h"
#include "Light/light.h"

#include "pbrt.h"

namespace CudaPBRT
{
	class Scene
	{
	public:
        Scene()
            : shapes(nullptr), materials(nullptr), lights(nullptr),
              shape_count(0), material_count(0), light_count(0)
        {
        }

        INLINE CPU_GPU bool Sample_Li(float rand, const glm::vec2& xi, const glm::vec3& p, const glm::vec3& normal, LightSample& sample)
        {
            ASSERT(light_count < 10);
            int light_id = static_cast<int>(glm::floor(rand * 10.f)) % light_count;
            sample = lights[light_id]->Sample_Li(p, normal, xi);
            
            sample.pdf /= light_count; // equivalent to divide by pdf 

            Intersection shadow_intersect;
            return (IntersectionNaive(sample.shadowRay, shadow_intersect) && shadow_intersect.isLight && shadow_intersect.id == light_id);
        }

        INLINE CPU_GPU float PDF_Li(int light_id, const glm::vec3& p, const glm::vec3& wiW, float t, const glm::vec3& normal)
        {
            return (lights[light_id]->PDF(p, wiW, t, normal) / static_cast<float>(light_count));
        }

        void FreeDataOnCuda()
        {
            FreeArrayOnCuda<Shape>(shapes, shape_count);
            FreeArrayOnCuda<Material>(materials, material_count);
            FreeArrayOnCuda<Light>(lights, light_count);
        }

		CPU_GPU bool IntersectionNaive(const Ray& ray, Intersection& intersection)
		{
            for (int i = 0; i < light_count; ++i)
            {
                Intersection it;
                if (lights[i]->IntersectionP(ray, it) && it < intersection)
                {
                    intersection = it;
                    intersection.isLight = true;
                    intersection.id = i;
                }
            }

            for (int i = 0; i < shape_count; ++i)
            {
                Intersection it;
                if (shapes[i]->IntersectionP(ray, it) && it < intersection)
                {
                    intersection = it;
                    intersection.id = i;
                    intersection.material_id = shapes[i]->material_id;
                }
            }
            return !(intersection.id < 0);
		}

	public:
        Shape** shapes; // shapes on device
        Material** materials; // materials on device
		Light** lights; // lights on device

        size_t shape_count;
        size_t material_count;
		size_t light_count;
	};
}
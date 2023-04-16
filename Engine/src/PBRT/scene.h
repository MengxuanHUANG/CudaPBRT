#pragma once

#include "pbrtDefine.h"
#include "intersection.h"
#include "Shape/shape.h"
#include "Material/material.h"
#include "Light/light.h"

#include "BVH/bvh.h"

#include "pbrt.h"

namespace CudaPBRT
{
	class Scene
	{
	public:
        Scene()
            : shapes(nullptr), materials(nullptr), lights(nullptr), vertices(nullptr),
              shape_count(0), material_count(0), light_count(0),
              boundings(nullptr), BVH(nullptr)
        {
        }

        INLINE CPU_GPU bool Sample_Li(float rand, const glm::vec2& xi, const glm::vec3& p, const glm::vec3& normal, LightSample& sample)
        {
            ASSERT(light_count < 10);
            int light_id = static_cast<int>(glm::floor(rand * 10.f)) % light_count;
            sample = lights[light_id]->Sample_Li(p, normal, xi);
            
            //sample.pdf;// /= light_count; // equivalent to divide by pdf 
            //printf("arae: %f\n", sample.pdf);
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
            CUDA_FREE(vertices);
            CUDA_FREE(boundings);
            CUDA_FREE(BVH);
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

            /*
            // BVH intersection
            int to_visit[64];
            int current_node = 0;
            int next_visit = 0;
            
            glm::vec3 invDir(glm::vec3(1.f) / ray.DIR);
            bool dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };

            while (true)
            {
                const BVHNode& node = BVH[current_node];
                const BoundingBox& bounding = boundings[node.boundingBoxId];
                
                //printf("Current: %d [%f, %f, %f] [%f, %f, %f]\n", current_node, 
                //    bounding.m_Min.x, bounding.m_Min.y, bounding.m_Min.z,
                //    bounding.m_Max.x, bounding.m_Max.y, bounding.m_Max.z);

                float t;

                if (bounding.IntersectP(ray, invDir, t))
                {
                    //printf("Current: %d\n", current_node);
                    if (node.primitiveId >= 0) // leaf node
                    {
                        Intersection it;
                        //printf("Test %s, primitive: %d\n", node.primitiveId < 12 ? "long box" : "short box", node.primitiveId);
                        if (t < intersection.t && shapes[node.primitiveId]->IntersectionP(ray, it) && it < intersection)
                        {
                            //printf("Hit %s, primitive: %d\n", node.primitiveId < 12 ? "long box" : "short box", node.primitiveId);
                            intersection = it;
                            intersection.id = node.primitiveId;
                            intersection.material_id = shapes[node.primitiveId]->material_id;
                        }
                        if (next_visit == 0) break;
                        current_node = to_visit[--next_visit];
                    }
                    else
                    {
                        current_node = node.next;
                        to_visit[next_visit++] = node.next + 1;
                    }
                }
                else
                {
                    if (next_visit == 0) break;
                    current_node = to_visit[--next_visit];
                }
            }
            */
            return !(intersection.id < 0);
		}

	public:
        Shape** shapes; // shapes on device
        Material** materials; // materials on device
		Light** lights; // lights on device
        glm::vec3* vertices;

        size_t shape_count;
        size_t material_count;
		size_t light_count;

        // BVH
        BoundingBox* boundings;
        BVHNode* BVH;
	};
}
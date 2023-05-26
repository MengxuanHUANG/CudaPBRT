#pragma once

#include "pbrtDefine.h"
#include "intersection.h"
#include "Shape/shape.h"
#include "Material/material.h"
#include "Light/light.h"

#include "BVH/bvh.h"

#include "pbrt.h"
#include "texture.h"

namespace CudaPBRT
{
    /*
    ** Scene data used on Cuda kernals
    */
	class GPUScene
	{
	public:
        GPUScene()
            : shapes(nullptr), materials(nullptr), lights(nullptr), 
              vertices(nullptr), normals(nullptr), uvs(nullptr),
              shape_count(0), material_count(0), light_count(0),
              boundings(nullptr), BVH(nullptr),
              envMap(0)
        {
        }

        INLINE CPU_GPU bool Sample_Li(float rand, const glm::vec2& xi, const glm::vec3& p, const glm::vec3& normal, LightSample& sample)
        {
            if (light_count <= 0.f)
            {
                return false;
            }
            ASSERT(light_count < 10);
            int light_id = static_cast<int>(glm::floor(rand * 10.f)) % light_count;
            sample = lights[light_id]->Sample_Li(p, normal, xi);
            
            Intersection shadow_intersect;
            return (SceneIntersection(sample.shadowRay, shadow_intersect) && shadow_intersect.isLight && shadow_intersect.id == light_id);
        }

        INLINE CPU_GPU float PDF_Li(int light_id, const glm::vec3& p, const glm::vec3& wiW, float t, const glm::vec3& normal)
        {
            return (lights[light_id]->PDF(p, wiW, t, normal) / static_cast<float>(light_count));
        }

        void FreeDataOnCuda()
        {
            printf("start free cuda\n");
            printf("start free arrays on cuda\n");
            FreeArrayOnCuda<Shape>(shapes, shape_count);
            CUDA_CHECK_ERROR();
            FreeArrayOnCuda<Material>(materials, material_count);
            CUDA_CHECK_ERROR();
            FreeArrayOnCuda<Light>(lights, light_count);
            printf("end free arrays on cuda\n");

            printf("start free BVH arrays on cuda\n");
            CUDA_FREE(vertices);
            CUDA_CHECK_ERROR();
            CUDA_FREE(normals);
            CUDA_CHECK_ERROR();
            CUDA_FREE(uvs);
            CUDA_CHECK_ERROR();
            CUDA_FREE(boundings);
            CUDA_CHECK_ERROR();
            CUDA_FREE(BVH);
            CUDA_CHECK_ERROR();
            printf("end free BVH arrays on cuda\n");

            printf("end free cuda\n");
        }

        CPU_GPU bool SceneIntersection(const Ray& ray, Intersection& intersection)
        {
#if USE_BVH 1
            return BVHIntersection(ray, intersection);
#else
            return IntersectionNaive(ray, intersection);
#endif
        }

		INLINE CPU_GPU bool IntersectionNaive(const Ray& ray, Intersection& intersection)
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
                    intersection.isLight = false;
                    intersection.material_id = shapes[i]->material_id;
                }
            }
            return !(intersection.id < 0);
		}

        INLINE CPU_GPU bool BVHIntersection(const Ray& ray, Intersection& intersection)
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

            // BVH intersection
            int to_visit[64];
            int current_node = 0;
            int next_visit = 0;

            glm::vec3 invDir(glm::vec3(1.f) / ray.DIR);
            bool dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };
            int max_it = 0;
            while (true)
            {
                const BVHNode& node = BVH[current_node];
                const BoundingBox& bounding = boundings[node.boundingBoxId];

                float t;

                if (bounding.IntersectP(ray, invDir, t) && t < intersection.t)
                {
                    if (node.primitiveId >= 0) // leaf node
                    {
                        Intersection it;
                        for (int i = 0; i < node.primitiveCount; ++i)
                        {
                            if (shapes[i + node.primitiveId]->IntersectionP(ray, it) && it < intersection)
                            {
                                intersection = it;
                                intersection.id = node.primitiveId;
                                intersection.isLight = false;
                                intersection.material_id = shapes[node.primitiveId + i]->material_id;
                            }
                        }
                        if (next_visit == 0) break;
                        current_node = to_visit[--next_visit];
                    }
                    else
                    {
                        current_node = dirIsNeg[node.splitAxis] ? node.next : node.next + 1;
                        to_visit[next_visit++] = dirIsNeg[node.splitAxis] ? node.next + 1 : node.next;
                    }
                }
                else
                {
                    if (next_visit == 0) break;
                    current_node = to_visit[--next_visit];
                }
            }

            return !(intersection.id < 0);
        }

	public:
        Shape** shapes; // shapes on device
        Material** materials; // materials on device
		Light** lights; // lights on device
        glm::vec3* vertices;
        glm::vec3* normals;
        glm::vec2* uvs;

        size_t shape_count;
        size_t material_count;
		size_t light_count;

        // BVH
        BoundingBox* boundings;
        BVHNode* BVH;
        EnvironmentMap envMap;
	};

    /*
    ** Scene data that will be maintenanced on CPU
    */
    class CPUScene
    {
    public:
        CPUScene() = default;
        ~CPUScene()
        {
            m_Textures.clear();
            shapeData.clear();
            materialData.clear();
            lightData.clear();
        }

    public:
        GPUScene m_GPUScene;

        // textures
        std::vector<uPtr<CudaTexture>> m_Textures;

        // shapes' data
        std::vector<ShapeData> shapeData;
        // materials' data
        std::vector<MaterialData> materialData;
        // lights' data
        std::vector<LightData> lightData;
    };
}
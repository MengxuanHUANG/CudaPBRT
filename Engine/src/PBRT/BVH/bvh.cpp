#include "bvh.h"

#include "PBRT/Shape/triangle.h"
#include "PBRT/pbrt.h"
#include <queue>
#include <format>

namespace CudaPBRT
{
	void CreateBVH(std::vector<ShapeData>& shapeData,
				   const std::vector<glm::vec3>& vertices,
				   std::vector<BoundingBox>& bounding_boxes,
				   std::vector<BVHNode>& BVH)
	{
		// stores the bounding boxes of primitives
		std::vector<std::pair<int, BoundingBox>> boxes;

		//TODO: reserve for bounding box vector
		
		// Create Bounding boxes for all shapes
		for (int i = 0; i < shapeData.size(); ++i)
		{
			boxes.emplace_back(i, Triangle::GetWorldBounding({ vertices[shapeData[i].verticeId[0]], 
															   vertices[shapeData[i].verticeId[1]], 
															   vertices[shapeData[i].verticeId[2]]}));

			//std::string tem = R"(AABB( min[{}, {}, {}], max[{}, {}, {}] ))";
			//
			//const BoundingBox& aabb = boxes.back().second;
			//
			//std::cout << std::vformat(tem, std::make_format_args(
			//	aabb.m_Min.x, aabb.m_Min.y, aabb.m_Min.z, 
			//	aabb.m_Max.x, aabb.m_Max.y, aabb.m_Max.z)) << std::endl;
		}

		std::queue<std::pair<int, int>> taskes;
		
		taskes.emplace(0, shapeData.size());

		while (!taskes.empty())
		{
			auto [start, end] = taskes.front();
			taskes.pop();

			//printf("[%d, %d]\n", start, end);
			//printf("%d taskes left\n", taskes.size());

			int nPrimitives = end - start;
			if (nPrimitives == 0)
			{
				continue;
			}

			// Check if this is an leaf node
			if (nPrimitives == 1)
			{
				// Create leaf node
				bounding_boxes.emplace_back(boxes[start].second);
				BVH.emplace_back(boxes[start].first, bounding_boxes.size() - 1);

				continue;
			}

			// 1. Compute unioned Bounding Box, and the centroid of the primitive set
			BoundingBox bounding;
			for (int i = start; i < end; ++i)
			{
				bounding = bounding.Union(boxes[i].second);
			}

			BoundingBox centroid_bound;
			for (int i = start; i < end; ++i)
			{
				centroid_bound = centroid_bound.Union(boxes[i].second.Centroid());
			}
			int dim = centroid_bound.MaximumExtent();
			int mid = (start + end) / 2;
			
			// 2. Split bounding box set
			//if (centroid_bound.m_Max[dim] == centroid_bound.m_Min[dim]) // the centroid bounds have zero volume, no need split
			//{
			//	for (int i = start; i < end; ++i) {
			//		bounding_boxes.emplace_back(boxes[i].second);
			//	}
			//	
			//	BVH.emplace_back(boxes[start].first, bounding_boxes.size() - 1, -1);
			//
			//	continue;
			//}
			// Add the bounding box
			bounding_boxes.emplace_back(bounding);
			BVH.emplace_back(-1, bounding_boxes.size() - 1, dim, bounding_boxes.size() + taskes.size());

			// Middle Split 
			float p_mid = 0.5f * (centroid_bound.m_Max[dim] + centroid_bound.m_Min[dim]);
			auto* midPtr = std::partition(&boxes[start], &boxes[end - 1] + 1,
					[dim, p_mid](const std::pair<int, BoundingBox>& pair) {
						return pair.second.Centroid()[dim] < p_mid;
					});
			mid = midPtr - &boxes[0];

			if (mid == start || mid == end)
			{
				// Equal split
				mid = (start + end) / 2;
				std::nth_element(&boxes[start], &boxes[mid], &boxes[end - 1] + 1,
					[dim](const std::pair<int, BoundingBox>& a, const std::pair<int, BoundingBox>& b) {
						return a.second.Centroid()[dim] < b.second.Centroid()[dim];
					});
			}
			//for (auto& pair : boxes)
			//{
			//	printf("%d ", pair.first);
			//}
			//printf("\n");

			// 3. Emplace taskes for left branch and right branch
			taskes.emplace(start, mid); // Left
			taskes.emplace(mid, end); // Right
		}
		PerspectiveCamera camera(680, 680, 19.5f, glm::vec3(0, 5.5, -30), glm::vec3(0, 2.5, 0));
		float t;
/*
		for (int w = 0; w < 680; w++)
		{
			for (int h = 0; h < 680; ++h)
			{
				if (w != 500 || h != 100) continue;
				//printf("[%d, %d]\n", w, h);
				Ray ray = CastRay(camera, {w, h});
				glm::vec3 invDir(glm::vec3(1.f) / ray.DIR);
				bool dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };

				int to_visit[64];
				int current_node = 0;
				int next_visit = 0;
				
				while (true)
				{
					const BVHNode& node = BVH[current_node];
					const BoundingBox& bounding = bounding_boxes[node.boundingBoxId];

					//printf("Current: %d [%f, %f, %f] [%f, %f, %f]\n", current_node, 
					//    bounding.m_Min.x, bounding.m_Min.y, bounding.m_Min.z,
					//    bounding.m_Max.x, bounding.m_Max.y, bounding.m_Max.z);

					float t0, t1;

					if (bounding.IntersectP(ray, invDir, t))
					{
						//printf("Current: %d\n", current_node);
						if (node.primitiveId >= 0) // leaf node
						{
							printf("Test %s, primitive: %d\n", node.primitiveId < 12 ? "long box" : "short box", node.primitiveId);

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

					//printf("To Visit: ");
					//for (int i = 0; i < next_visit; ++i)
					//{
					//	printf("%d ", to_visit[i]);
					//}
					//printf("\n");
				}
			}
		}

		*/
	}
}
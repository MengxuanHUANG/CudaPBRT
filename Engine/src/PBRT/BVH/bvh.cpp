#include "bvh.h"

#include "PBRT/Shape/triangle.h"
#include "PBRT/Shape/square.h"
#include "PBRT/Shape/cube.h"
#include "PBRT/Shape/sphere.h"

#include "PBRT/pbrt.h"
#include <queue>
#include <format>

namespace CudaPBRT
{
	INLINE BoundingBox CreateBoundingBox(const ShapeData& data, const std::vector<glm::vec3>& vertices)
	{
		switch (data.type)
		{
		case ShapeType::Triangle:
			return Triangle::GetWorldBounding({ vertices[data.triangle.vId[0]],
												vertices[data.triangle.vId[1]],
												vertices[data.triangle.vId[2]] });
		case ShapeType::Square:
			return Square::GetWorldBounding(data);
		case ShapeType::Cube:
			return Cube::GetWorldBounding(data);
		case ShapeType::Sphere:
			return Sphere::GetWorldBounding(data);
		default:
			printf("Unknown ShapeType!\n");
		}

		return {};
	}

	void CreateBVH(std::vector<ShapeData>& shapeData,
		const std::vector<glm::vec3>& vertices,
		std::vector<BoundingBox>& bounding_boxes,
		std::vector<BVHNode>& BVH)
	{
		// store the bounding boxes of primitives
		std::vector<std::pair<int, BoundingBox>> aabbs;
		// store the ordered primitives
		std::vector<ShapeData> ordered_primitives;
		//TODO: reserve for bounding box vector

		// Create Bounding boxes for all shapes
		for (int i = 0; i < shapeData.size(); ++i)
		{
			aabbs.emplace_back(i, CreateBoundingBox(shapeData[i], vertices));
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
				bounding_boxes.emplace_back(aabbs[start].second);
				ordered_primitives.emplace_back(shapeData[aabbs[start].first]);
				BVH.emplace_back(ordered_primitives.size() - 1, 1, bounding_boxes.size() - 1);

				continue;
			}

			// 1. Compute unioned Bounding Box, and the centroid of the primitive set
			BoundingBox bounding;
			for (int i = start; i < end; ++i)
			{
				bounding.Union(aabbs[i].second);
			}

			BoundingBox centroid_bound;
			for (int i = start; i < end; ++i)
			{
				centroid_bound.Union(aabbs[i].second.Centroid());
			}
			int dim = centroid_bound.MaximumExtent();
			int mid = (start + end) / 2;

			// Add the bounding box
			bounding_boxes.emplace_back(bounding);

			// 2. Split bounding box set
			// if the centroid bounds have zero volume, no need split
			if (centroid_bound.m_Max[dim] == centroid_bound.m_Min[dim])
			{
				for (int i = start; i < end; ++i) {
					ordered_primitives.emplace_back(shapeData[aabbs[i].first]);
				}

				BVH.emplace_back(ordered_primitives.size() - nPrimitives, nPrimitives, bounding_boxes.size() - 1);
				continue;
			}

#if BVH_SAH
			// SAH split
			if (nPrimitives <= 4)
			{
				// Equal split
				mid = (start + end) / 2;
				std::nth_element(&aabbs[start], &aabbs[mid], &aabbs[end - 1] + 1,
					[dim](const std::pair<int, BoundingBox>& a, const std::pair<int, BoundingBox>& b) {
						return a.second.Centroid()[dim] < b.second.Centroid()[dim];
					});
			}
			else
			{
				// Allocate bucketInfo for SAH partition buckets
				constexpr int nBuckets = 12; // try 11 possible partitions
				struct BucketInfo {
					int count = 0;
					BoundingBox bounds;
				};
				BucketInfo buckets[nBuckets];

				// initialize BucketInfo
				float dim_min = centroid_bound.m_Min[dim];
				float dim_max = centroid_bound.m_Max[dim];
				for (int i = start; i < end; ++i)
				{
					int b = nBuckets * (aabbs[i].second.Centroid()[dim] - dim_min) / (dim_max - dim_min);
					b = glm::clamp(b, 0, nBuckets - 1);
					buckets[b].count++;
					buckets[b].bounds.Union(aabbs[i].second);
				}

				// compute cost for splitting after each bucket & find min cost
				float min_cost = FloatMax;
				int min_cost_bucket = 0;
				float costs[nBuckets];
				for (int i = 0; i < nBuckets - 1; ++i)
				{
					BoundingBox b0, b1;
					int count0 = 0, count1 = 0;

					for (int j = 0; j <= i; ++j)
					{
						b0.Union(buckets[j].bounds);
						count0 += buckets[j].count;
					}

					for (int j = i + 1; j < nBuckets; ++j) 
					{
						b1.Union(buckets[j].bounds);
						count1 += buckets[j].count;
					}
					costs[i] = 0.125f * (count0 * b0.SurfaceArea() + count1 * b1.SurfaceArea()) / bounding.SurfaceArea();
					if (costs[i] < min_cost)
					{
						min_cost = costs[i];
						min_cost_bucket = i;
					}
				}

				// create leaf node or split
				if (nPrimitives > 255 || min_cost < nPrimitives) // split
				{
					auto* midPtr = std::partition(&aabbs[start], &aabbs[end - 1] + 1,
						[=](const std::pair<int, BoundingBox>& pair) {
							int b = nBuckets * (pair.second.Centroid()[dim] - dim_min) / (dim_max - dim_min);
							b = glm::clamp(b, 0, nBuckets - 1);
							return b <= min_cost_bucket;
						});
					mid = midPtr - aabbs.data();
				}
				else // create leaf node
				{
					for (int i = start; i < end; ++i) {
						ordered_primitives.emplace_back(shapeData[aabbs[i].first]);
					}

					BVH.emplace_back(ordered_primitives.size() - nPrimitives, nPrimitives, bounding_boxes.size() - 1);
					continue;
				}
			}
#else
			// Middle Split 
			float p_mid = 0.5f * (centroid_bound.m_Max[dim] + centroid_bound.m_Min[dim]);
			auto* midPtr = std::partition(&aabbs[start], &aabbs[end - 1] + 1,
				[dim, p_mid](const std::pair<int, BoundingBox>& pair) {
					return pair.second.Centroid()[dim] < p_mid;
				});
			mid = midPtr - &aabbs[0];

			if (mid == start || mid == end)
			{
				// Equal split
				mid = (start + end) / 2;
				std::nth_element(&aabbs[start], &aabbs[mid], &aabbs[end - 1] + 1,
					[dim](const std::pair<int, BoundingBox>& a, const std::pair<int, BoundingBox>& b) {
						return a.second.Centroid()[dim] < b.second.Centroid()[dim];
					});
			}
#endif
			// 3. Emplace taskes for left branch and right branch
			BVH.emplace_back(-1, 0, bounding_boxes.size() - 1, dim, bounding_boxes.size() + taskes.size());
			taskes.emplace(start, mid); // Left
			taskes.emplace(mid, end); // Right
		}

		shapeData.swap(ordered_primitives);

		//PerspectiveCamera camera(680, 680, 19.5f, glm::vec3(0, 5.5, -30), glm::vec3(0, 2.5, 0));
		//
		//int w = 350;
		//int h = 350;
		//Ray ray = CastRay(camera, { w, h });
		//glm::vec3 invDir(glm::vec3(1.f) / ray.DIR);
		//bool dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };
		//
		//int to_visit[64];
		//int current_node = 0;
		//int next_visit = 0;
		//
		//float closest_t = FloatMax;
	/*
		while (true)
		{
			const BVHNode& node = BVH[current_node];
			const BoundingBox& bounding = bounding_boxes[node.boundingBoxId];

			float t;
			if (bounding.IntersectP(ray, invDir, t) && t < closest_t)
			{
				closest_t = t;
				//printf("Current: %d\n", current_node);
				if (node.primitiveId >= 0) // leaf node
				{
					Intersection it;
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
			std::cout << "current: " << current_node;
			if (next_visit > 0)
			{
				std::cout << ", next: ";
				for (int i = 0; i < next_visit; ++i)
				{
					std::cout << to_visit[i] << " ";
				}
				std::cout << std::endl;
			}
			else
			{
				std::cout << ", next: empty" << std::endl;
			}
		}
	*/
	}
}
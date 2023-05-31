#pragma once

#include "PBRT/pbrtDefine.h"
#include "PBRT/intersection.h"
#include "PBRT/ray.h"
#include "PBRT/BVH/boundingBox.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

namespace CudaPBRT
{
	struct TriangleData
	{
		glm::ivec3 vId	= glm::ivec3(-1);
		glm::ivec3 nId	= glm::ivec3(-1);
		glm::ivec3 uvId = glm::ivec3(-1);
		
		CPU_GPU TriangleData() 
		{}
		
		CPU_GPU TriangleData(const glm::ivec3& v_id, 
							 const glm::ivec3& n_id = glm::ivec3(-1),
							 const glm::ivec3& uv_id = glm::ivec3(-1))
			:vId(v_id), nId(n_id), uvId(uv_id)
		{}
	};

	enum class ShapeType : unsigned char
	{
		None = 0,
		Sphere,
		Cube,
		Triangle,
		Square,
	};

	ShapeType Str2ShapeType(const char* str);

	struct ShapeData
	{
		ShapeType type = ShapeType::None;
		int material_id = -1;

		glm::vec3 translation = glm::vec3(0.f);
		glm::vec3 rotation = glm::vec3(0.f);
		glm::vec3 scale = glm::vec3(1.f);

		TriangleData triangle;

		glm::vec3* vertices = nullptr;
		glm::vec3* normals = nullptr;
		glm::vec2* uvs = nullptr;

		ShapeData(const int& material_id,
				  const TriangleData& tri,
				  glm::vec3* vertices,
				  glm::vec3* normal = nullptr,
				  glm::vec2* uv = nullptr)
			: type(ShapeType::Triangle), material_id(material_id), 
			  triangle(tri), vertices(vertices), normals(normal), uvs(uv)
		{}

		ShapeData(ShapeType type,
				  const int& material_id,
				  const glm::vec3& translate = glm::vec3(0.f),
				  const glm::vec3 rotate = glm::vec3(0.f),
				  const glm::vec3& scale = glm::vec3(1.f))
			:type(type), material_id(material_id), translation(translate), rotation(rotate), scale(scale)
		{}
	};

	class Shape
	{
	public:
		CPU_GPU Shape(const ShapeData& data)
			:material_id(data.material_id)
		{}

		CPU_GPU virtual glm::vec3 GetNormal(const glm::vec3& p) const = 0;
		CPU_GPU virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const = 0;
		CPU_GPU virtual float SimpleIntersection(const Ray& ray) const = 0;

		CPU_GPU virtual float Area() const { return 0.f; }
		CPU_GPU virtual glm::vec3 Sample(const glm::vec2& xi) const { return glm::vec3(0.f); }

	public:
		INLINE CPU_GPU static void ComputeTransforms(const glm::vec3& translate,
													 const glm::vec3& rotate,
													 const glm::vec3& scale,
													 glm::mat4& transform, 
													 glm::mat4& invTransform, 
													 glm::mat3& TransposeInvTransform)
		{
			Shape::ComputeTransform(translate, rotate, scale, transform);

			invTransform = glm::inverse(transform);
			TransposeInvTransform = glm::transpose(glm::inverse(glm::mat3(transform)));
		}

		INLINE CPU_GPU static void ComputeTransform(const glm::vec3& translate,
													 const glm::vec3& rotate,
													 const glm::vec3& scale,
													 glm::mat4& transform)
		{
			glm::mat4 T = glm::translate(glm::mat4(1.f), translate);

			glm::mat4 S = glm::scale(glm::mat4(1.f), scale);

			glm::mat4 Rx = glm::rotate(glm::mat4(1.f), glm::radians(rotate.x), { 1.f, 0.f, 0.f });
			glm::mat4 Ry = glm::rotate(glm::mat4(1.f), glm::radians(rotate.y), { 0.f, 1.f, 0.f });
			glm::mat4 Rz = glm::rotate(glm::mat4(1.f), glm::radians(rotate.z), { 0.f, 0.f, 1.f });

			transform = T * Rx * Ry * Rz * S;
		}
	public:
		int material_id;
	};
}
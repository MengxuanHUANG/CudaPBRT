#pragma once

#include "PBRT/pbrtDefine.h"
#include "PBRT/intersection.h"
#include "PBRT/ray.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

namespace CudaPBRT
{
	enum class ShapeType
	{
		None = 0,
		Sphere,
		Cube,
		Triangle,
		Square,
	};

	struct ShapeData
	{
		ShapeType type = ShapeType::None;
		int material_id = -1;

		glm::vec3 translation = glm::vec3(0.f);
		glm::vec3 scale = glm::vec3(1.f);
		glm::vec3 rotation = glm::vec3(0.f);

		ShapeData(ShapeType type,
				  const int& material_id,
				  const glm::vec3& translate = glm::vec3(0.f),
				  const glm::vec3 scale = glm::vec3(1.f),
				  const glm::vec3& rotate = glm::vec3(0.f))
			:type(type), material_id(material_id), translation(translate), scale(scale), rotation(rotate)
		{}
	};

	class Shape
	{
	public:
		CPU_GPU Shape(const ShapeData& data)
			:material_id(data.material_id), shapeData(data)
		{
			glm::mat4 T = glm::translate(glm::mat4(1.f), data.translation);

			glm::mat4 S = glm::scale(glm::mat4(1.f), data.scale);

			glm::mat4 Rx = glm::rotate(glm::mat4(1.f), glm::radians(data.rotation.x), { 1.f, 0.f, 0.f });
			glm::mat4 Ry = glm::rotate(glm::mat4(1.f), glm::radians(data.rotation.y), { 0.f, 1.f, 0.f });
			glm::mat4 Rz = glm::rotate(glm::mat4(1.f), glm::radians(data.rotation.z), { 0.f, 0.f, 1.f });

			m_Transform = T * Rx * Ry * Rz * S;
			m_TransformInv = glm::inverse(m_Transform);
			m_TransformInvTranspose = glm::transpose(glm::inverse(glm::mat3(m_Transform)));
		}

		CPU_GPU virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const = 0;
		CPU_GPU virtual float Area() const { return 0.f; }
		CPU_GPU virtual glm::vec3 Sample(const glm::vec2& xi) const { return glm::vec3(0.f); }

	public:
		int material_id;
		glm::mat4 m_Transform;
		glm::mat4 m_TransformInv;
		glm::mat3 m_TransformInvTranspose;

	protected:
		ShapeData shapeData;
	};
}
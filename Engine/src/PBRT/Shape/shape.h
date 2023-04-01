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
		glm::vec3 translation = glm::vec3(0.f);
		glm::vec3 scale = glm::vec3(1.f);
		glm::vec3 rotation = glm::vec3(0.f);
	};

	class Shape
	{
	public:
		GPU_ONLY Shape(const ShapeData& data)
		{
			glm::mat4 T = glm::translate(glm::mat4(1.f), data.translation);

			glm::mat4 S = glm::scale(glm::mat4(1.f), data.scale);

			glm::mat4 Rx = glm::rotate(glm::mat4(1.f), data.rotation.x, { 1.f, 0.f, 0.f });
			glm::mat4 Ry = glm::rotate(glm::mat4(1.f), data.rotation.y, { 0.f, 1.f, 0.f });
			glm::mat4 Rz = glm::rotate(glm::mat4(1.f), data.rotation.z, { 0.f, 0.f, 1.f });

			m_Transform = T * Rx * Ry * Rz * S;
			m_TransformInv = glm::inverse(m_Transform);
			m_TransformInvTranspose = glm::transpose(glm::inverse(glm::mat3(m_Transform)));
		}

		GPU_ONLY virtual bool IntersectionP(const Ray& ray, Intersection& intersection) const = 0;

	protected:
		glm::mat4 m_Transform;
		glm::mat4 m_TransformInv;
		glm::mat3 m_TransformInvTranspose;
	};
}
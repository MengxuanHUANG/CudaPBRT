#include "Camera.h"

#include <glm/gtx/transform.hpp>

namespace CudaPBRT
{

	PerspectiveCamera::PerspectiveCamera(unsigned int w,
										 unsigned int h,
										 const glm::vec3& pos,
										 const glm::vec3& forward,
										 const glm::vec3& worldUp,
										 float lenRadius,
										 float focalDistance)
		:fovy(45.f),
		 width(w),
		 height(h),
		 position(pos),
		 forward(forward),
		 worldUp(worldUp),
		 lensRadius(lenRadius),
		 focalDistance(focalDistance)
	{}

	PerspectiveCamera::PerspectiveCamera(const PerspectiveCamera& c)
		:fovy(c.fovy),
		 width(c.width),
		 height(c.height),
		 position(c.position),
		 forward(c.forward),
		 up(c.up),
		 right(c.right),
		worldUp(c.worldUp)
	{}

	void PerspectiveCamera::RecomputeAttributes()
	{
		// recompute right, up
		right = glm::normalize(glm::cross(forward, worldUp));
		up = glm::normalize(glm::cross(right, forward));
	}

	void PerspectiveCamera::RotateAboutUp(float deg)
	{
		glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), deg, up);
		
		glm::vec4 ref = glm::vec4(forward, 1.f);
		ref = rotation * ref;

		forward = glm::normalize(glm::vec3(ref));
		
		RecomputeAttributes();
	}

	void PerspectiveCamera::RotateAboutRight(float deg)
	{
		glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), deg, right);

		glm::vec4 ref = glm::vec4(forward, 1.f);
		ref = rotation * ref;

		forward = glm::normalize(glm::vec3(ref));

		RecomputeAttributes();
	}

	void PerspectiveCamera::RotateTheta(float deg)
	{
		glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), deg, right);
		
		glm::vec3 ref = position + 10.f * forward;
		position = position - ref;

		position = glm::vec3(rotation * glm::vec4(position, 1.f));
		position = position + ref;
		RecomputeAttributes();
	}

	void PerspectiveCamera::RotatePhi(float deg)
	{
		glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), deg, up);

		glm::vec3 ref = position + 10.f * forward;
		position = position - ref;

		position = glm::vec3(rotation * glm::vec4(position, 1.f));
		position = position + ref;
		RecomputeAttributes();
	}

	void PerspectiveCamera::TranslateAlongLook(float amt)
	{
		glm::vec3 trans = forward * amt;
		position += trans;
	}

	void PerspectiveCamera::TranslateAlongRight(float amt)
	{
		glm::vec3 translation = right * amt;
		position += translation;
	}

	void PerspectiveCamera::TranslateAlongUp(float amt)
	{
		glm::vec3 translation = up * amt;
		position += translation;
	}

	void PerspectiveCamera::Zoom(float amt)
	{
		TranslateAlongLook(amt);
	}
}
#include "Camera.h"



namespace CudaPBRT
{

	PerspectiveCamera::PerspectiveCamera(unsigned int w,
										 unsigned int h,
										 const glm::vec3& pos,
										 const glm::vec3& ref,
										 const glm::vec3& worldUp,
										 float lenRadius,
										 float focalDistance)
		:fovy(45.f),
		 width(w),
		 height(h),
		 position(pos),
		 ref(ref),
		 worldUp(worldUp),
		 lensRadius(lenRadius),
		 focalDistance(focalDistance)
	{
		RecomputeAttributes();
	}

	PerspectiveCamera::PerspectiveCamera(const PerspectiveCamera& c)
		:fovy(c.fovy),
		 width(c.width),
		 height(c.height),
		 position(c.position),
		 ref(c.ref),
		worldUp(c.worldUp)
	{
		RecomputeAttributes();
	}

	void PerspectiveCamera::RecomputeAttributes()
	{
		// recompute right, up
		forward = glm::normalize(ref - position);
		right = glm::normalize(glm::cross(forward, worldUp));
		up = glm::normalize(glm::cross(right, forward));
	}
}
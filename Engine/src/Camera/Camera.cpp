#include "Camera.h"



namespace CudaPBRT
{

	PerspectiveCamera::PerspectiveCamera(unsigned int w,
										 unsigned int h,
										 float fovy,
										 const glm::vec3& pos,
										 const glm::vec3& ref,
										 const glm::vec3& worldUp,
										 float lenRadius,
										 float focalDistance)
		:width(w),
		 height(h),
		 fovy(fovy),
		 position(pos),
		 ref(ref),
		 worldUp(worldUp),
		 lensRadius(lenRadius),
		 focalDistance(focalDistance)
	{
		RecomputeAttributes();
	}

	PerspectiveCamera::PerspectiveCamera(const PerspectiveCamera& c)
		:width(c.width),
		 height(c.height),
	 	 fovy(c.fovy),
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
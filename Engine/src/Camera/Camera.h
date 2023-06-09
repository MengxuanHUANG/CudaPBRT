#pragma once

#include "Math/Matrix.h"

namespace CudaPBRT
{
	class PerspectiveCamera
	{
	public:
		PerspectiveCamera(unsigned int w, 
						  unsigned int h, 
						  float fovy = 45.f,
						  const glm::vec3& pos = glm::vec3(0, 0, -30),
						  const glm::vec3& ref = glm::vec3(0, 0, 0),
						  const glm::vec3& worldUp = glm::vec3(0, 1, 0),
						  float lenRadius = 0.f,
						  float focalDistance = 1.f);
		PerspectiveCamera(const PerspectiveCamera& c);

		void RecomputeAttributes();

	public:
		// perspective camera parameters
		unsigned int width, height;
		float fovy;

		float aspect;

		glm::vec3 position; // camera's position in world space
		glm::vec3 ref;

		glm::vec3 forward;
		glm::vec3 up;
		glm::vec3 right;
		glm::vec3 worldUp;

		// len camera parameters
		float lensRadius, focalDistance;
	};
}
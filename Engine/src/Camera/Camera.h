#pragma once

#include "Math/Matrix.h"

namespace CudaPBRT
{
	class PerspectiveCamera
	{
	public:
		PerspectiveCamera(unsigned int w, 
						  unsigned int h, 
						  const glm::vec3& pos = glm::vec3(0, 0, -30),
						  const glm::vec3& forward = glm::vec3(0, 0, 1),
						  const glm::vec3& worldUp = glm::vec3(0, 1, 0),
						  float lenRadius = 0.f,
						  float focalDistance = 0.f);
		PerspectiveCamera(const PerspectiveCamera& c);

		void RecomputeAttributes();

		void RotateAboutUp(float deg);
		void RotateAboutRight(float deg);

		void RotateTheta(float deg);
		void RotatePhi(float deg);

		void TranslateAlongLook(float amt);
		void TranslateAlongRight(float amt);
		void TranslateAlongUp(float amt);

		void Zoom(float amt);

	public:
		// perspective camera parameters
		float fovy;
		unsigned int width, height;

		float aspect;

		glm::vec3 position; // camera's position in world space
		glm::vec3 forward;
		glm::vec3 up;
		glm::vec3 right;
		glm::vec3 worldUp;

		// len camera parameters
		float lensRadius, focalDistance;
	};
}
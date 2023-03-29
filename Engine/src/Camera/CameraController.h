#pragma once

#include "Camera.h"
#include "Window/Events/Event.h"
#include "Window/MouseProperty.h"

namespace CudaPBRT
{
	class PerspectiveCameraController
	{
	public:
		PerspectiveCameraController(PerspectiveCamera& camera, 
									float panSpeed = 0.05f, 
									float zoomSpeed = 0.02f, 
									float rotateSpeed = 0.02f);

		virtual bool OnEvent(Event& event);

	protected:
		bool OnMouseMoved(MouseMovedEvent&);

		void RotateAboutUp(float deg);
		void RotateAboutRight(float deg);

		void RotateTheta(float deg);
		void RotatePhi(float deg);

		void TranslateAlongLook(float amt);
		void TranslateAlongRight(float amt);
		void TranslateAlongUp(float amt);

		void Zoom(float amt);
	
	public:
		// camera control parameters
		float panSpeed;
		float zoomSpeed;
		float rotateSpeed;

	protected:
		glm::vec2 m_MousePosPre;
		PerspectiveCamera& m_Camera;
	};
}
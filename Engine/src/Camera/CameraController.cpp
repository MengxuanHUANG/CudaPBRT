#include "CameraController.h"

#include "Window/Events/EventDispatcher.h"
#include "Core/Application.h"
#include "Window/Window.h"
#include "Window/KeyCode.h"

#include <iostream>
#include <format>

namespace CudaPBRT
{
	PerspectiveCameraController::PerspectiveCameraController(PerspectiveCamera& camera, float panSpeed, float zoomSpeed, float rotateSpeed)
		: m_Camera(camera)
	{}

	bool PerspectiveCameraController::OnEvent(Event& event)
	{
		EventDispatcher dispatcher(event);
		dispatcher.Dispatch<MouseMovedEvent>(std::bind(&PerspectiveCameraController::OnMouseMoved, this, std::placeholders::_1));
		return false;
	}

	bool PerspectiveCameraController::OnMouseMoved(MouseMovedEvent& event)
	{
		Window* window = Application::GetApplication().GetWindow();
		glm::vec2 newPos = glm::vec2(event.GetX(), event.GetY());
		glm::vec2 offset = newPos - m_MousePosPre;

		if (window->GetKeyButtonState(MY_KEY_LEFT_ALT))
		{
			if (window->GetMouseButtonState(MY_MOUSE_BN_LEFT))
			{
				// Rotation
				glm::vec2 diff = rotateSpeed * offset;

				// inverse rotation to obtain normal result
				m_Camera.RotatePhi(-diff.x); 
				m_Camera.RotateTheta(-diff.y);
			}
			else if (window->GetMouseButtonState(MY_MOUSE_BN_MIDDLE))
			{
				// Panning
				glm::vec2 diff = panSpeed * offset;
				m_Camera.TranslateAlongRight(-diff.x); // inverse x panning to obtain normal result
				m_Camera.TranslateAlongUp(diff.y);
			}
			else if (window->GetMouseButtonState(MY_MOUSE_BN_RIGHT))
			{
				// Zoom
				glm::vec2 diff = zoomSpeed * offset;
				m_Camera.Zoom(diff.y);
			}
		}

		m_MousePosPre = glm::vec2(event.GetX(), event.GetY());
		return false;
	}
}
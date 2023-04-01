#include "CameraController.h"

#include "Window/Events/EventDispatcher.h"
#include "Core/Application.h"
#include "Window/Window.h"
#include "Window/KeyCode.h"

#include <iostream>
#include <format>
#include <glm/gtx/transform.hpp>

namespace CudaPBRT
{
	PerspectiveCameraController::PerspectiveCameraController(PerspectiveCamera& camera, float panSpeed, float zoomSpeed, float rotateSpeed)
		: m_Camera(camera), 
		  panSpeed(panSpeed), 
		  zoomSpeed(zoomSpeed), 
		  rotateSpeed(rotateSpeed)
	{}

	bool PerspectiveCameraController::OnEvent(Event& event)
	{
		EventDispatcher dispatcher(event);
		return dispatcher.Dispatch<MouseMovedEvent>(std::bind(&PerspectiveCameraController::OnMouseMoved, this, std::placeholders::_1));
	}

	void PerspectiveCameraController::RotateAboutUp(float deg)
	{
		glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), deg, m_Camera.up);

		glm::vec4 ref = glm::vec4(m_Camera.forward, 1.f);
		ref = rotation * ref;

		m_Camera.ref -= m_Camera.position;
		m_Camera.ref = rotation * glm::vec4(m_Camera.ref, 1.f);
		m_Camera.ref += m_Camera.position;

		m_Camera.RecomputeAttributes();
	}

	void PerspectiveCameraController::RotateAboutRight(float deg)
	{
		glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), deg, m_Camera.right);

		m_Camera.ref -= m_Camera.position;
		m_Camera.ref = rotation * glm::vec4(m_Camera.ref, 1.f);
		m_Camera.ref += m_Camera.position;

		m_Camera.RecomputeAttributes();
	}

	void PerspectiveCameraController::RotateTheta(float deg)
	{
		glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), deg, m_Camera.right);

		m_Camera.position -= m_Camera.ref;

		m_Camera.position = glm::vec3(rotation * glm::vec4(m_Camera.position, 1.f));
		m_Camera.position += m_Camera.ref;
		m_Camera.RecomputeAttributes();
	}
	void PerspectiveCameraController::RotatePhi(float deg)
	{
		glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), deg, m_Camera.up);

		m_Camera.position -= m_Camera.ref;

		m_Camera.position = glm::vec3(rotation * glm::vec4(m_Camera.position, 1.f));
		m_Camera.position += m_Camera.ref;
		m_Camera.RecomputeAttributes();
	}

	void PerspectiveCameraController::TranslateAlongLook(float amt)
	{
		glm::vec3 trans = m_Camera.forward * amt;
		m_Camera.position += trans;
		m_Camera.ref += trans;
	}

	void PerspectiveCameraController::TranslateAlongRight(float amt)
	{
		glm::vec3 trans = m_Camera.right * amt;
		m_Camera.position += trans;
		m_Camera.ref += trans;
	}

	void PerspectiveCameraController::TranslateAlongUp(float amt)
	{
		glm::vec3 trans = m_Camera.up * amt;
		m_Camera.position += trans;
		m_Camera.ref += trans;
	}

	void PerspectiveCameraController::Zoom(float amt)
	{
		TranslateAlongLook(amt);
	}

	bool PerspectiveCameraController::OnMouseMoved(MouseMovedEvent& event)
	{
		Window* window = Application::GetApplication().GetWindow();
		glm::vec2 newPos = glm::vec2(event.GetX(), event.GetY());
		glm::vec2 offset = newPos - m_MousePosPre;
		m_MousePosPre = newPos;

		if (window->GetKeyButtonState(MY_KEY_LEFT_ALT))
		{
			if (window->GetMouseButtonState(MY_MOUSE_BN_LEFT))
			{
				// Rotation
				glm::vec2 diff = rotateSpeed * offset;
				// inverse rotation to obtain normal result
				RotatePhi(-diff.x); 
				RotateTheta(-diff.y);

				return true;
			}
			else if (window->GetMouseButtonState(MY_MOUSE_BN_MIDDLE))
			{
				// Panning
				glm::vec2 diff = panSpeed * offset;
				TranslateAlongRight(-diff.x); // inverse x panning to obtain normal result
				TranslateAlongUp(diff.y);

				return true;
			}
			else if (window->GetMouseButtonState(MY_MOUSE_BN_RIGHT))
			{
				// Zoom
				glm::vec2 diff = zoomSpeed * offset;
				Zoom(diff.y);

				return true;
			}
		}
		return false;
	}
}
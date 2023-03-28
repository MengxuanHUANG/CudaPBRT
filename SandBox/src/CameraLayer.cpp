#include "CameraLayer.h"
#include "Core/Application.h"
#include "Window/Window.h"


CameraLayer::CameraLayer(const std::string& name)
	:Layer(name)
{
	WindowProps* props = Application::GetApplication().GetWindow()->GetWindowProps();
	m_Camera = mkU<PerspectiveCamera>(props->width, props->height);
	m_CamController = mkU<PerspectiveCameraController>(*m_Camera);
}

bool CameraLayer::OnEvent(Event& event)
{
	return m_CamController->OnEvent(event);
}
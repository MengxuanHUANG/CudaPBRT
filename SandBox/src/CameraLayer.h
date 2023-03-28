#pragma once

#include "Core/Layer.h"
#include "Camera/Camera.h"
#include "Camera/CameraController.h"

using namespace CudaPBRT;

class CameraLayer : public Layer
{
public:
	CameraLayer(const std::string& name);

	virtual bool OnEvent(Event& event) override;

public:
	uPtr<PerspectiveCamera> m_Camera;
	uPtr<PerspectiveCameraController> m_CamController;
};
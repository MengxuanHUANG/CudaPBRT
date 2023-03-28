#pragma once

#include "CameraLayer.h"

#include "Core/Application.h"

class SandBox : public Application
{
public:
	SandBox()
	{
		PushLayer(new CameraLayer("Camera Layer"));
	}
};
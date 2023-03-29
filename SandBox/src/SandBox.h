#pragma once

#include "TestLayer.h"

#include "Core/Application.h"

class SandBox : public Application
{
public:
	SandBox(const WindowProps& props)
		:Application(props)
	{
		PushLayer(new TestLayer("Test Layer"));
	}
};
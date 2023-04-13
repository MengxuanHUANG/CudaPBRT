#pragma once

#include "Core/Layer.h"
#include "Window/Window.h"
#include "Camera/Camera.h"
#include "Camera/CameraController.h"

#include "PBRT/scene.h"

using namespace CudaPBRT;

namespace CudaPBRT
{
	class CudaPathTracer;
}

class TestLayer : public Layer
{
public:
	TestLayer(const std::string& name);
	~TestLayer();

	virtual void OnAttach() override;
	virtual void OnDetach() override;

	virtual void OnUpdate(float delatTime) override;
	virtual void OnImGuiRendered(float deltaTime) override;

	virtual bool OnEvent(Event& event) override;

protected:
	bool OnWindowResize(WindowResizeEvent& event);

	void AddCornellBox_Triangles(std::vector<ShapeData>& shapeData, int material_a, int material_b);

public:
	uPtr<PerspectiveCamera> m_Camera;
	uPtr<PerspectiveCameraController> m_CamController;

protected:
	Window* window;
	uPtr<Scene> m_Scene;
	uPtr<CudaPathTracer> m_CudaPBRT;
};
#pragma once

#include "Core/Layer.h"

using namespace CudaPBRT;

namespace CudaPBRT
{
	class CudaPathTracer;
	class CudaTexture;
	class Window;
	class CPUScene;
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

protected:
	Window* window; // only a reference to the window
	uPtr<CPUScene> m_Scene;
	uPtr<CudaPathTracer> m_CudaPBRT;

	int m_SelectedMaterial;

	float m_FrameTime;

	std::string m_CurrentFile;

	static std::string JSON_PATH;
	static std::string IMG_SAVE_PATH;
};
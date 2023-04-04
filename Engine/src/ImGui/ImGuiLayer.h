#pragma once

#include "Core/Layer.h"

struct ImGuiIO;

namespace CudaPBRT
{
	class ImGuiLayer : public Layer
	{
	public:
		ImGuiLayer(const std::string& name = "ImGuiLayer");

		virtual void OnAttach() override;
		virtual void OnDetach() override;

		// should be editor only function
		virtual void OnImGuiRendered(float) override;

		void Begin();
		void End();

	protected:
		ImGuiIO* io;
	};
}
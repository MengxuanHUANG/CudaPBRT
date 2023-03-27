#pragma once

#include "Window/Events/Event.h"

#include <string>

namespace CudaPBRT
{
	class Layer
	{
	public:
		Layer(const std::string& name)
			: m_LayerName(name)
		{}
		virtual ~Layer() = default;

		virtual void OnAttach() {}
		virtual void OnDetach() {}
		virtual void OnUpdate(float) {}
		virtual bool OnEvent(Event&) { return false; }

		// should be editor only function
		virtual void OnImGuiRendered(float) {};

	protected:
		std::string m_LayerName;
	};
}
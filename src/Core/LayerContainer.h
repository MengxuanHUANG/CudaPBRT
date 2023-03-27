#pragma once

#include "Core.h"
#include "Window/Events/Event.h"

#include <vector>

namespace CudaPBRT
{
	class Layer;

	class LayerStack
	{
	public:
		~LayerStack();

		void PushLayer(Layer* layer);
		void PopLayer(Layer* layer, bool deleteLayer = false);

		std::vector<Layer*>::iterator begin() { return m_LayerStack.begin(); }
		std::vector<Layer*>::iterator end() { return m_LayerStack.end(); }
	protected:
		std::vector<Layer*> m_LayerStack;
	};
}
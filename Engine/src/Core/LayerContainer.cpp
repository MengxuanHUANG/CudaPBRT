#include "LayerContainer.h"
#include "Layer.h"

namespace CudaPBRT
{
	LayerStack::~LayerStack()
	{
		for (Layer* layer : m_LayerStack)
		{
			delete layer;
		}

		m_LayerStack.clear();
	}

	void LayerStack::PushLayer(Layer* layer)
	{
		ASSERT(layer);
		m_LayerStack.emplace_back(layer);
	}

	void LayerStack::PopLayer(Layer* layer, bool deleteLayer)
	{
		ASSERT(layer);
		auto it = std::find(m_LayerStack.begin(), m_LayerStack.end(), layer);

		if (it != m_LayerStack.end())
		{
			m_LayerStack.erase(it);
		}

		if (deleteLayer)
		{
			delete layer;
		}
	}
}
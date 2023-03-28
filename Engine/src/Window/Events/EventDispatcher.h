#pragma once

#include "Event.h"
#include <functional>

namespace CudaPBRT
{
	class EventDispatcher
	{
	public:
		template<typename T>
		using EventCallBackFn = std::function<bool(T&)>;

		EventDispatcher(Event& event)
			:m_Event(event)
		{}

		template<typename T>
		bool Dispatch(EventCallBackFn<T> fn)
		{
			if (T::GetStaticType() == m_Event.GetEventType())
			{
				m_Event.m_Handled = fn(*(T*)&m_Event);
				return true;
			}
			else
			{
				return false;
			}
		}

	protected:
		Event& m_Event;
	};
}
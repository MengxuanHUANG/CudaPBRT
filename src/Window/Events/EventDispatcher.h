#pragma once

#include "Event.h"
#include <functional>

namespace CudaPBRT
{
	class EventDispatcher
	{
	public:
		typedef std::function<bool(Event&)> EventCallBackFn;
		EventDispatcher(Event& event)
			:m_Event(event)
		{}

		template<typename T>
		bool Dispatch(EventCallBackFn fn)
		{
			if (T::GetStaticType() == m_Event.GetEventType())
			{
				return fn(m_Event);
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
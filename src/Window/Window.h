#pragma once
#include "Core/Core.h"
#include "Events/Event.h"

#include <string>
#include <functional>

namespace CudaPBRT
{
	struct WindowProps
	{
		unsigned int width;
		unsigned int height;
		std::string title;

		WindowProps(const std::string& title = "New Window",
					unsigned int width = 2048,
					unsigned height = 1080)
			: width(width), height(height), title(title)
		{
		}
	};

	class Window
	{
	public:
		typedef std::function<bool(Event&)> EventCallbackFn;

		virtual ~Window() = default;
	
		virtual void OnUpdate() = 0;

		// Be careful to reinterpret_cast the returned pointer
		virtual void* GetNativeWindow() const = 0;

	public:
		static uPtr<Window> CreateWindow(EventCallbackFn fn, const WindowProps& props = WindowProps());
	};
}
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

		WindowProps(unsigned int width = 2048,
					unsigned height = 1080,
					const std::string& title = "New Window")
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
		virtual double GetTime() = 0;

		// Be careful to reinterpret_cast the returned pointer
		virtual void* GetNativeWindow() const = 0;
		virtual WindowProps* GetWindowProps() = 0;

		// Get input states
		virtual int GetMouseButtonState(int button) { return 0; }
		virtual int GetKeyButtonState(int keycode) { return 0; }
		virtual std::pair<double, double> GetCursorPosition() { return { 0, 0 }; }

	public:
		static uPtr<Window> CreateWindow(EventCallbackFn fn, const WindowProps& props = WindowProps());
	};
}
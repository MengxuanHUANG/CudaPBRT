#include "Event.h"
#include "Window/KeyCode.h"

#include <string>
#include <format>

namespace CudaPBRT
{
	WindowResizeEvent::WindowResizeEvent(unsigned int width, unsigned int height)
		:m_Size(width, height)
	{}

	std::string WindowResizeEvent::ToString() const
	{
		auto& [w, h] = m_Size;
		return std::vformat("WindowResizeEvent: [w: {}, h: {}]", std::make_format_args(w, h));
	}

	MouseMovedEvent::MouseMovedEvent(float x, float y)
		:m_Move(x, y)
	{}

	std::string MouseMovedEvent::ToString() const
	{
		auto& [x, y] = m_Move;
		return std::vformat("MouseMovedEvent: [x: {}, y: {}]", std::make_format_args(x, y));
	}

	MouseScrolledEvent::MouseScrolledEvent(float x, float y)
		:m_Offset(x, y)
	{}

	std::string MouseScrolledEvent::ToString() const
	{
		auto& [x, y] = m_Offset;
		return std::vformat("MouseScrolledEvent: [OffsetX: {}, OffsetY: {}]", std::make_format_args(x, y));
	}

	MouseButtonPressedEvent::MouseButtonPressedEvent(int button)
		:MouseButtonEvent(button)
	{}

	std::string MouseButtonPressedEvent::ToString() const
	{
		return std::vformat("Mouse({})Button Pressed", std::make_format_args(m_Button));
	}

	MouseButtonReleasedEvent::MouseButtonReleasedEvent(int button)
		:MouseButtonEvent(button)
	{}

	std::string MouseButtonReleasedEvent::ToString() const
	{
		return std::vformat("Mouse({})Button Released", std::make_format_args(m_Button));
	}

	KeyPressedEvent::KeyPressedEvent(int keycode, int repeatCount)
		: KeyBoardEvent(keycode), m_RepeatCount(repeatCount)
	{}

	std::string KeyPressedEvent::ToString() const
	{
		
		return std::vformat("Key: ({}) Pressed", std::make_format_args(KeyMap.at(m_Keycode)));
	}

	std::string KeyReleasedEvent::ToString() const
	{
		return std::vformat("Key: ({}) Released", std::make_format_args(KeyMap.at(m_Keycode)));
	}

	std::string KeyTypedEvent::ToString() const
	{
		return std::vformat("Key: ({}) Typed", std::make_format_args(KeyMap.at(m_Keycode)));
	}
}
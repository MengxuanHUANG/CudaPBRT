#pragma once

#include "Core/Core.h"
#include <string>

namespace CudaPBRT
{
	enum class EventType
	{
		None = 0,
		WindowClose, WindowResize,
		KeyPressed, KeyTyped, KeyReleased,
		MouseMoved, MouseClicked, MouseScrolled, MouseButtonPressed, MouseButtonReleased
	};

	enum EventCategory
	{
		None = 0,
		EventCategoryApplication	= BIT(0),
		EventCategoryInput			= BIT(1),
		EventCategoryKeyboard		= BIT(2),
		EventCategoryMouse			= BIT(3),
		EventCategoryMouseButton	= BIT(4)
	};

	class Event
	{
	public:
		virtual EventType GetEventType() const = 0;
		virtual const char* GetName() const = 0;
		virtual int GetCategoryFlags() const = 0;

		virtual std::string ToString() const { return GetName(); }

		inline bool IsInCategory(EventCategory category) const { return GetCategoryFlags() & category; }

		bool m_Handled = false;
	};

#define DECLARE_EVENT(type) static EventType GetStaticType() { return EventType::##type; }\
							virtual EventType GetEventType() const override { return GetStaticType(); }\
							virtual const char* GetName() const override { return #type; }

#define DECLARE_EVENT_FLAG(category) virtual int GetCategoryFlags() const override { return category; }


	// Application Events begin
	class WindowCloseEvent : public Event
	{
	public:
		WindowCloseEvent() {}

		DECLARE_EVENT(WindowClose)
		DECLARE_EVENT_FLAG(EventCategoryApplication)
	};
	class WindowResizeEvent : public Event
	{
	public:
		WindowResizeEvent(unsigned int width, unsigned int height);
		std::string ToString() const override;

		inline unsigned int GetWidth() const { return m_Size.first; }
		inline unsigned int GetHeight() const { return m_Size.second; }

		DECLARE_EVENT(WindowResize)
		DECLARE_EVENT_FLAG(EventCategoryApplication)
	private:
		std::pair<unsigned int, unsigned int> m_Size;
	};
	// Application Events end

	// Mouse Input Events begin
	class MouseMovedEvent : public Event
	{
	public:
		MouseMovedEvent(float x, float y);

		std::string ToString() const override;

		inline float GetX() const { return m_Move.first; }
		inline float GetY()	const { return m_Move.second; }

		DECLARE_EVENT(MouseMoved)
		DECLARE_EVENT_FLAG(EventCategoryMouse | EventCategoryInput)

	private:
		std::pair<float, float> m_Move;
	};
	class MouseButtonEvent : public Event
	{
	public:
		inline int GetMouseButton() const { return m_Button; }

		DECLARE_EVENT_FLAG(EventCategoryMouse | EventCategoryInput)
	protected:
		MouseButtonEvent(int button)
			:m_Button(button) {}
		int m_Button;
	};
	class MouseScrolledEvent : public Event
	{
	public:
		MouseScrolledEvent(float x, float y);

		std::string ToString() const override;
		inline float GetOffsetX() const { return m_Offset.first; }
		inline float GetOffsetY() const { return m_Offset.second; }

		DECLARE_EVENT(MouseScrolled)
		DECLARE_EVENT_FLAG(EventCategoryMouse | EventCategoryInput)

	private:
		std::pair<float, float> m_Offset;
	};
	class MouseButtonPressedEvent : public MouseButtonEvent
	{
	public:
		MouseButtonPressedEvent(int button);

		std::string ToString()const override;

		DECLARE_EVENT(MouseButtonPressed)
	};
	class MouseButtonReleasedEvent : public MouseButtonEvent
	{
	public:
		MouseButtonReleasedEvent(int button);

		std::string ToString()const override;

		DECLARE_EVENT(MouseButtonReleased)
	};
	// Mouse Input Events end
	 
	// Keyboard Input Events begin
	class KeyBoardEvent :public Event
	{
	public:
		inline int GetKeycode() const { return m_Keycode; }

		DECLARE_EVENT_FLAG(EventCategoryKeyboard | EventCategoryInput)

	protected:
		KeyBoardEvent(int keycode)
			:m_Keycode(keycode) {}
		int m_Keycode;
	};

	class KeyPressedEvent :public KeyBoardEvent
	{
	public:
		KeyPressedEvent(int keycode, int repeatCount);

		inline int GetRepeatCount() const { return m_RepeatCount; }

		std::string ToString() const override;
		DECLARE_EVENT(KeyPressed);
	private:
		int m_RepeatCount;
	};

	class KeyReleasedEvent : public KeyBoardEvent
	{
	public:
		KeyReleasedEvent(int keycode)
			:KeyBoardEvent(keycode) {}

		std::string ToString() const override;

		DECLARE_EVENT(KeyReleased)
	};

	class KeyTypedEvent : public KeyBoardEvent
	{
	public:
		KeyTypedEvent(int keycode)
			:KeyBoardEvent(keycode) {}

		std::string ToString() const override;

		DECLARE_EVENT(KeyTyped)
	};
	// Keyboard Input Events end
}
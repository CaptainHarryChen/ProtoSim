#pragma once

namespace viewer {

enum EventType
{
    FramebufferSize,
    MouseButton,
    Cursor,
    Scroll,
    Key
};

struct FramebufferSizeEvent
{
    int width;
    int height;
};

struct MouseButtonEvent
{
    int button;
    int action;
    int mods;
};

struct CursorEvent
{
    double xpos;
    double ypos;
};

struct ScrollEvent
{
    double xoffset;
    double yoffset;
};

struct KeyEvent
{
    int key;
    int scancode;
    int action;
    int mods;
};

struct Event
{
    EventType type;
    union
    {
        FramebufferSizeEvent framebuffer_size;
        MouseButtonEvent     mouse_button;
        CursorEvent          cursor;
        ScrollEvent          scroll;
        KeyEvent             key;
    };
};

}  // namespace viewer

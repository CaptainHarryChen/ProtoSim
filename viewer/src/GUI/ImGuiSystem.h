#pragma once
#include <GLFW/glfw3.h>

class ImGuiSystem
{
public:
    ImGuiSystem(GLFWwindow *window);
    virtual ~ImGuiSystem();

    void BeforeRender();
    void AfterRender();
};

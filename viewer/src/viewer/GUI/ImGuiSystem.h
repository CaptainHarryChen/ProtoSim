#pragma once
#include <GLFW/glfw3.h>

namespace viewer {

class ImGuiSystem
{
public:
    ImGuiSystem(GLFWwindow* window);
    virtual ~ImGuiSystem();

    void BeforeRender();
    void AfterRender();
};

}  // namespace viewer

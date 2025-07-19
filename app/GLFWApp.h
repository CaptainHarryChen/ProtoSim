#pragma once
#include <memory>
#include <string>
#include <vector>
#include <Event/Event.h>

class RenderSystem;
class ImGuiSystem;
class Camera;
class Object;
struct GLFWwindow;

class GLFWApp
{
public:
    static std::shared_ptr<GLFWApp> GetInstance();
    static std::shared_ptr<GLFWApp> GetInstance(const std::string &name, int init_width, int init_height);

    GLFWApp(const std::string &name, int init_width, int init_height);
    virtual ~GLFWApp();

    bool RunOneFrame();
    void Run();

    void SetCamera(std::shared_ptr<Camera> camera);
    void AddObject(std::shared_ptr<Object> object);
    std::shared_ptr<RenderSystem> GetRenderSystem();

public:
    std::vector<std::shared_ptr<Object>> m_objects;

protected:
    bool m_has_initialized = false;
    double m_last_time = -1.0;

    GLFWwindow *m_window;
    std::shared_ptr<RenderSystem> m_render_system;
    std::shared_ptr<ImGuiSystem> m_imgui_system;

    void ProcessEvent(const Event &event);

private:
    static void framebuffer_size_callback(GLFWwindow *window, int width, int height);
    static void mousebutton_callback(GLFWwindow *window, int button, int action, int mods);
    static void cursor_callback(GLFWwindow *window, double xpos, double ypos);
    static void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
};

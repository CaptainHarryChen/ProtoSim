#include "GLFWApp.h"
#include <mutex>
#include <cassert>
#include <imgui/imgui.h>
#include <Render/RenderSystem.h>
#include <GUI/ImGuiSystem.h>
#include <Object/Object.h>
#include <Camera/Camera.h>

static std::shared_ptr<GLFWApp> g_main_class_ptr = nullptr;
static std::once_flag g_main_class_flag;

std::shared_ptr<GLFWApp> GLFWApp::GetInstance()
{
    assert(g_main_class_ptr != nullptr);
    return g_main_class_ptr;
}

std::shared_ptr<GLFWApp> GLFWApp::GetInstance(const std::string &name, int init_width, int init_height)
{
    std::call_once(g_main_class_flag, [&]
                   { g_main_class_ptr = std::shared_ptr<GLFWApp>(new GLFWApp(name, init_width, init_height)); });
    return g_main_class_ptr;
}

GLFWApp::GLFWApp(const std::string &name, int init_width, int init_height)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    m_window = glfwCreateWindow(init_width, init_height, name.c_str(), nullptr, nullptr);
    glfwSetWindowUserPointer(m_window, this);
    glfwMakeContextCurrent(m_window);
    glfwSetFramebufferSizeCallback(m_window, GLFWApp::framebuffer_size_callback);
    glfwSetScrollCallback(m_window, GLFWApp::scroll_callback);
    glfwSetMouseButtonCallback(m_window, GLFWApp::mousebutton_callback);
    glfwSetCursorPosCallback(m_window, GLFWApp::cursor_callback);
    glfwSetKeyCallback(m_window, GLFWApp::key_callback);

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    m_render_system = std::make_shared<RenderSystem>();
    m_render_system->m_viewport_width = init_width;
    m_render_system->m_viewport_height = init_height;
    m_imgui_system = std::make_shared<ImGuiSystem>(m_window);

    m_has_initialized = true;
}

GLFWApp::~GLFWApp()
{
    if (m_has_initialized)
    {
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }
}

bool GLFWApp::RunOneFrame()
{
    assert(m_has_initialized);
    
    double current_time = glfwGetTime();
    double delta_time = current_time - m_last_time;
    if (m_last_time < 0.0)
        delta_time = 0.0;
    m_last_time = current_time;

    glfwPollEvents();

    for (auto &object : m_objects)
    {
        object->Update(delta_time);
    }

    m_imgui_system->BeforeRender();
    m_render_system->RenderOneFrame();
    m_imgui_system->AfterRender();

    glfwSwapBuffers(m_window);
    return !glfwWindowShouldClose(m_window);
}

void GLFWApp::Run()
{
    for (;;)
    {
        if (!RunOneFrame())
            break;
    }
}

void GLFWApp::SetCamera(std::shared_ptr<Camera> camera)
{
    glfwGetWindowSize(m_window, &camera->viewport_width, &camera->viewport_height);
    m_render_system->SetCamera(camera);
}

void GLFWApp::AddObject(std::shared_ptr<Object> object)
{
    m_objects.push_back(object);
}

std::shared_ptr<RenderSystem> GLFWApp::GetRenderSystem()
{
    return m_render_system;
}

void GLFWApp::ProcessEvent(const Event &event)
{
    // ImGui has the highest priority
    switch (event.type)
    {
    case EventType::FramebufferSize:
        m_render_system->m_viewport_height = event.framebuffer_size.height;
        m_render_system->m_viewport_width = event.framebuffer_size.width;
        break;
    case EventType::MouseButton:
        if (ImGui::GetIO().WantCaptureMouse)
            return;
        break;
    case EventType::Cursor:
        if (ImGui::GetIO().WantCaptureMouse)
            return;
        break;
    case EventType::Scroll:
        if (ImGui::GetIO().WantCaptureMouse)
            return;
        break;
    case EventType::Key:
        if (ImGui::GetIO().WantCaptureKeyboard)
            return;
        if (event.key.action == GLFW_PRESS && event.key.key == GLFW_KEY_ESCAPE)
            glfwSetWindowShouldClose(m_window, true);
        break;
    }
    for (auto &object : m_objects)
    {
        object->ProcessEvent(event);
    }
}

void GLFWApp::framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    auto app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
    app->ProcessEvent({.type = EventType::FramebufferSize, .framebuffer_size = {width, height}});
}
void GLFWApp::mousebutton_callback(GLFWwindow *window, int button, int action, int mods)
{
    auto app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
    app->ProcessEvent({.type = EventType::MouseButton, .mouse_button = {button, action, mods}});
}
void GLFWApp::cursor_callback(GLFWwindow *window, double xpos, double ypos)
{
    auto app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
    app->ProcessEvent({.type = EventType::Cursor, .cursor = {xpos, ypos}});
}
void GLFWApp::scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    auto app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
    app->ProcessEvent({.type = EventType::Scroll, .scroll = {xoffset, yoffset}});
}
void GLFWApp::key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    auto app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
    app->ProcessEvent({.type = EventType::Key, .key = {key, scancode, action, mods}});
}

#include "RenderSystem.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <Mesh/Mesh.h>
#include <Render/Light.h>

static std::shared_ptr<RenderSystem> g_main_class_ptr = nullptr;
static std::once_flag g_main_class_flag;

std::shared_ptr<RenderSystem> RenderSystem::GetInstance()
{
    std::call_once(g_main_class_flag, [&]
                   { g_main_class_ptr = std::shared_ptr<RenderSystem>(new RenderSystem()); });
    return g_main_class_ptr;
}

RenderSystem::RenderSystem(std::string name, int init_width, int init_height)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    m_window = glfwCreateWindow(init_width, init_height, name.c_str(), nullptr, nullptr);
    glfwMakeContextCurrent(m_window);
    glfwSetFramebufferSizeCallback(m_window, OrbitControl::framebuffer_size_callback);
    glfwSetScrollCallback(m_window, OrbitControl::scroll_callback);
    glfwSetMouseButtonCallback(m_window, OrbitControl::mousebutton_callback);
    glfwSetCursorPosCallback(m_window, OrbitControl::cursor_callback);

    m_camera = std::make_shared<OrbitControl>(m_window);

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

RenderSystem::~RenderSystem()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(m_window);
    glfwTerminate();
}

void RenderSystem::AddLight(std::shared_ptr<Light> light)
{
    m_lights.push_back(light);
}

void RenderSystem::AddRenderObject(std::shared_ptr<RenderObject> render_object)
{
    m_render_objects.push_back(render_object);
}

bool RenderSystem::ProcessControl()
{
    if (glfwWindowShouldClose(m_window))
        return false;
    glfwPollEvents();
    m_camera->processInput(m_window);
    return true;
}

void RenderSystem::RenderOneFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    std::vector<LightInfo> light_infos;
    std::vector<ShadowMappingInfo> shadow_mapping_infos;
    for (auto &light : m_lights)
    {
        light_infos.push_back(light->GetLightInfo());
        shadow_mapping_infos.push_back(light->CreateShadowMappingInfo(m_render_objects));
    }

    int width, height;
    glfwGetWindowSize(m_window, &width, &height);
    glViewport(0, 0, width, height);
    glClearColor(m_clear_color.x, m_clear_color.y, m_clear_color.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    CameraInfo camera_info;
    m_camera->computeMVP(camera_info.view, camera_info.projection);
    camera_info.view_pos = m_camera->getPos();
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    camera_info.viewport = {1.0f * viewport[0], 1.0f * viewport[1], 1.0f * viewport[2], 1.0f * viewport[3]};

    for (auto &object : m_render_objects)
    {
        object->Draw(camera_info, light_infos, shadow_mapping_infos);
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(m_window);
}

#include "OrbitCameraRenderer.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <Mesh/Mesh.h>
#include <Render/Light.h>

static std::shared_ptr<OrbitCameraRenderer> g_main_class_ptr = nullptr;
static std::once_flag g_main_class_flag;

std::shared_ptr<OrbitCameraRenderer> OrbitCameraRenderer::GetInstance()
{
    std::call_once(g_main_class_flag, [&]
                   { g_main_class_ptr = std::shared_ptr<OrbitCameraRenderer>(new OrbitCameraRenderer()); });
    return g_main_class_ptr;
}

OrbitCameraRenderer::OrbitCameraRenderer(std::string name, int init_width, int init_height)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    window = glfwCreateWindow(init_width, init_height, name.c_str(), nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, OrbitControl::framebuffer_size_callback);
    glfwSetScrollCallback(window, OrbitControl::scroll_callback);
    glfwSetMouseButtonCallback(window, OrbitControl::mousebutton_callback);
    glfwSetCursorPosCallback(window, OrbitControl::cursor_callback);

    camera = std::make_shared<OrbitControl>(window);

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

OrbitCameraRenderer::~OrbitCameraRenderer()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

bool OrbitCameraRenderer::ProcessControl()
{
    if (glfwWindowShouldClose(window))
        return false;
    glfwPollEvents();
    camera->processInput(window);
    return true;
}

void OrbitCameraRenderer::RenderOneFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    std::vector<LightInfo> light_infos;
    std::vector<ShadowMappingInfo> shadow_mapping_infos;
    for (auto &light : lights)
    {
        light_infos.push_back(light->GetLightInfo());
        shadow_mapping_infos.push_back(light->CreateShadowMappingInfo(render_objects));
    }

    int width, height;
    glfwGetWindowSize(window, &width, &height);
    glViewport(0, 0, width, height);
    glClearColor(clearColor.x, clearColor.y, clearColor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    CameraInfo camera_info;
    camera->computeMVP(camera_info.view, camera_info.projection);
    camera_info.viewPos = camera->getPos();
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    camera_info.viewport = {1.0f * viewport[0], 1.0f * viewport[1], 1.0f * viewport[2], 1.0f * viewport[3]};

    for (auto &object : render_objects)
    {
        object->Draw(camera_info, light_infos, shadow_mapping_infos);
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
}

#include "OrbitCameraRenderer.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <Mesh/SurfaceMesh.h>
#include <Object/CubeLight.h>
#include <Geometry/Sphere.h>
#include <Geometry/LineSegment.h>

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
    return false;
}

void OrbitCameraRenderer::RenderOneFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    int width, height;
    glfwGetWindowSize(window, &width, &height);
    glViewport(0, 0, width, height);
    glClearColor(clearColor.x, clearColor.y, clearColor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glm::mat4 model, view, projection;
    camera->computeMVP(model, view, projection);

    // Set uniform variables about lights
    std::vector<glm::vec3> lightPoses;
    std::vector<glm::vec3> lightColors;
    std::vector<unsigned int> depthMapIDs;
    std::vector<bool> lightsOn;
    float far_plane = cubelights[0]->getFarPlane();
    for (auto &light : cubelights)
    {
        if (light->IsOn())
            light->generateShadowMap(meshes);
        depthMapIDs.push_back(light->getDepthMap());
        lightPoses.push_back(light->getPos());
        lightColors.push_back(light->getColor());
    }
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    glm::vec4 __viewport = {1.0f * viewport[0], 1.0f * viewport[1], 1.0f * viewport[2], 1.0f * viewport[3]};

    for (auto &light : cubelights)
    {
        if (!light->IsOn())
            continue;
        light->getMesh()->Draw(
            model, view, projection,
            light->getPos(), light->getColor(),
            0.2f);
    }

    for (auto &mesh : meshes)
    {
        mesh->Draw(
            depthMapIDs,
            model, view, projection,
            lightPoses, lightColors,
            camera->getPos(),
            lightsOn,
            far_plane,
            true);
    }

    for (auto &sphere : spheres)
    {
        sphere->Draw(
            model, view, projection,
            lightPoses, lightColors, lightsOn,
            __viewport, camera->getPos());
    }

    for (auto &line : lines)
    {
        line->Draw(model, view, projection, glm::vec3(1.0f), lightsOn[0] || lightsOn[1] || lightsOn[2] || lightsOn[3]);
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
}

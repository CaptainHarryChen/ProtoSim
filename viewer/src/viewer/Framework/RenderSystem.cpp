#include "RenderSystem.h"
#include <glad/glad.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <viewer/RenderObject/Mesh.h>
#include <viewer/Framework/Light.h>
#include <viewer/Framework/Camera.h>

namespace viewer {

RenderSystem::RenderSystem()
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
}

RenderSystem::~RenderSystem()
{
}

void RenderSystem::SetCamera(std::shared_ptr<Camera> camera)
{
    m_camera = camera;
}

void RenderSystem::AddLight(std::shared_ptr<Light> light)
{
    m_lights.push_back(light);
}

void RenderSystem::AddRenderObject(std::shared_ptr<RenderObject> render_object)
{
    m_render_objects.push_back(render_object);
}

void RenderSystem::RenderOneFrame()
{
    std::vector<LightInfo>         light_infos;
    std::vector<ShadowMappingInfo> shadow_mapping_infos;
    for (auto& light : m_lights)
    {
        light_infos.push_back(light->GetLightInfo());
        shadow_mapping_infos.push_back(light->CreateShadowMappingInfo(m_render_objects));
    }

    CameraInfo camera_info = m_camera->GetCameraInfo();

    glClearColor(m_clear_color.x, m_clear_color.y, m_clear_color.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, m_viewport_width, m_viewport_height);

    for (auto& object : m_render_objects)
    {
        object->Draw(camera_info, light_infos, shadow_mapping_infos);
    }
}

}  // namespace viewer

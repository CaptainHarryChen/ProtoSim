#include "LightScene.h"
#include <string>
#include <imgui.h>
#include <viewer/Object/CubeLight.h>
#include <viewer/Framework/RenderSystem.h>
#include <viewer/GLFWApp.h>

namespace viewer {

void LightSceneControlGUI::Draw(const CameraInfo& camera_info, const std::vector<LightInfo>& light_infos, const std::vector<ShadowMappingInfo>& shadow_mapping_infos)
{
    ImGui::Begin("Control Panel");
    for (size_t i = 0; i < m_light_num; ++i)
    {
        if (ImGui::CollapsingHeader(("light#" + std::to_string(i)).c_str()))
        {
            ImGui::Checkbox(("light#" + std::to_string(i) + " on / off").c_str(), ( bool* )&m_light_on[i]);
            ImGui::ColorEdit3(("light#" + std::to_string(i) + " color").c_str(), ( float* )(&m_light_color[i]));
            ImGui::DragFloat(("light#" + std::to_string(i) + " position.x").c_str(), &m_light_pos[i].x, 0.05f);
            ImGui::DragFloat(("light#" + std::to_string(i) + " position.y").c_str(), &m_light_pos[i].y, 0.05f);
            ImGui::DragFloat(("light#" + std::to_string(i) + " position.z").c_str(), &m_light_pos[i].z, 0.05f);
        }
    }
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();
}

LightScene::LightScene()
{
    m_control_gui = std::make_shared<LightSceneControlGUI>();
    for (size_t i = 0; i < m_control_gui->m_light_num; ++i)
    {
        m_lights.push_back(std::make_shared<CubeLight>(m_control_gui->m_light_pos[i], m_control_gui->m_light_color[i]));
    }

    std::shared_ptr<RenderSystem> render_system = GLFWApp::GetInstance()->GetRenderSystem();
    for (auto& light : m_lights)
    {
        render_system->AddLight(light->GetLight());
        render_system->AddRenderObject(light->GetMesh());
    }
    render_system->AddRenderObject(m_control_gui);
}

void LightScene::Update(double delta_time)
{
    for (size_t i = 0; i < m_control_gui->m_light_num; ++i)
    {
        m_lights[i]->SetPosition(m_control_gui->m_light_pos[i]);
        m_lights[i]->SetColor(m_control_gui->m_light_color[i]);
        m_lights[i]->SetLightOn(m_control_gui->m_light_on[i]);
    }
}

}  // namespace viewer

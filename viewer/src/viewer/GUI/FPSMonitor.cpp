#include "FPSMonitor.h"
#include <chrono>
#include <imgui.h>
#include <viewer/Framework/Camera.h>
#include <viewer/Framework/Light.h>
#include <viewer/Framework/RenderSystem.h>

namespace viewer {

void FPSMonitor::Draw(const CameraInfo& camera_info, const std::vector<LightInfo>& light_infos, const std::vector<ShadowMappingInfo>& shadow_mapping_infos)
{
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->WorkPos);
    ImGui::SetNextWindowBgAlpha(0.4f);
    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;

    if (ImGui::Begin("Simulation FPS", nullptr, flags))
    {
        ImGuiIO& io = ImGui::GetIO();
        ImGui::Text("Physics simulation average %.3f ms/frame (%.1f FPS)", m_avg_time / 1000.0, m_fps);
    }
    ImGui::End();
}

void FPSMonitor::UpdateFPS(double elapsed_time)
{
    double sum_time = m_avg_time * m_time_history.size();
    m_time_history.push_back(elapsed_time);
    sum_time += elapsed_time;
    if (m_time_history.size() > m_history_size)
    {
        sum_time -= m_time_history.front();
        m_time_history.erase(m_time_history.begin());
    }
    m_avg_time = sum_time / m_time_history.size();
    m_fps      = 1000000.0 / m_avg_time;
}

}  // namespace viewer

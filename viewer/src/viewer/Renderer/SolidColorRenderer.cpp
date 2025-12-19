#include "SolidColorRenderer.h"
#include <glad/glad.h>
#include <viewer/Framework/Shader.h>
#include <viewer/RenderObject/Mesh.h>
#include <viewer/Framework/Camera.h>

namespace viewer {

SolidColorRenderer::SolidColorRenderer(const glm::vec3& color, bool only_edge)
    : m_color(color), m_only_edge(only_edge)
{
    m_shader = std::make_shared<Shader>("solid_color", false);
}

void SolidColorRenderer::Draw(const CameraInfo& camera, const std::vector<LightInfo>& light_infos, const std::vector<ShadowMappingInfo>& shadow_mapping_infos, const RenderObject* object) const
{
    if (m_only_edge)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    m_shader->use();
    m_shader->setMat4("model", object->GetModelMatrix());
    m_shader->setMat4("view", camera.view);
    m_shader->setMat4("projection", camera.projection);
    m_shader->setVec3("color", m_color);
    object->DrawVAO();
    if (m_only_edge)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void SolidColorRenderer::SetColor(const glm::vec3& color)
{
    this->m_color = color;
}

}  // namespace viewer

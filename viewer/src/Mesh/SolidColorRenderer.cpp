#include "SolidColorRenderer.h"
#include <glad/glad.h>
#include <Render/Shader.h>
#include <Mesh/Mesh.h>

SolidColorRenderer::SolidColorRenderer(const glm::vec3 &color, bool only_edge)
    : color(color), m_only_edge(only_edge)
{
    shader = std::make_shared<Shader>("solid_color", false);
}

void SolidColorRenderer::Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos, const RenderObject *object)
{
    if (m_only_edge)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    shader->use();
    shader->setMat4("model", object->model);
    shader->setMat4("view", camera.view);
    shader->setMat4("projection", camera.projection);
    shader->setVec3("color", color);
    object->DrawVAO();
    if (m_only_edge)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void SolidColorRenderer::SetColor(const glm::vec3 &color)
{
    this->color = color;
}

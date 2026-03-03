#include "CapsuleRenderer.h"
#include <glad/glad.h>
#include <viewer/Framework/Shader.h>
#include <viewer/Framework/RenderObject.h>
#include <viewer/Framework/Camera.h>

namespace viewer
{

CapsuleRenderer::CapsuleRenderer(const glm::vec2 &material, float radius, float half_height)
    : m_material(material), m_radius(radius), m_half_height(half_height)
{
    m_shader = std::make_shared<Shader>("capsule_raycast", true);
}

void CapsuleRenderer::Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos,
                           const std::vector<ShadowMappingInfo> &shadow_mapping_infos,
                           const RenderObject *object) const
{
    assert(light_infos.size() <= CapsuleRenderer::MAX_LIGHTS);

    m_shader->use();
    m_shader->setMat4("model", object->GetModelMatrix());
    m_shader->setMat4("view", camera.view);

    m_shader->setMat4("u_projMatrix", camera.projection);
    m_shader->setMat4("u_invProjMatrix", glm::inverse(camera.projection));
    m_shader->setVec4("u_viewport", camera.viewport);
    m_shader->setVec3("viewPos", camera.view_pos);
    m_shader->setFloat("u_radius", m_radius);
    m_shader->setFloat("u_halfHeight", m_half_height);

    for (size_t i = 0; i < light_infos.size(); ++i)
    {
        m_shader->setVec3("lightPos[" + std::to_string(i) + "]", light_infos[i].pos);
        m_shader->setVec3("lightColor[" + std::to_string(i) + "]", light_infos[i].color);
        m_shader->setBool("lightsOn[" + std::to_string(i) + "]", light_infos[i].is_on);
    }
    m_shader->setFloat("metalicIn", m_material.x);
    m_shader->setFloat("roughnessIn", m_material.y);

    object->DrawVAO();
}

void CapsuleRenderer::SetMaterial(const glm::vec2 &material)
{
    m_material = material;
}

}

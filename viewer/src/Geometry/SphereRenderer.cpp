#include "SphereRenderer.h"
#include <glad/glad.h>
#include <Render/Shader.h>
#include <Render/RenderObject.h>

SphereRenderer::SphereRenderer(const glm::vec2 &material, float radius) : m_material(material), m_radius(radius)
{
    m_shader = std::make_shared<Shader>("sphere_raycast", true);
}

void SphereRenderer::Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos, const RenderObject *object)
{
    assert(light_infos.size() <= SphereRenderer::MAX_LIGHTS);

    m_shader->use();
    m_shader->setMat4("model", object->m_model_mat);
    m_shader->setMat4("view", camera.view);

    m_shader->setMat4("u_projMatrix", camera.projection);
    m_shader->setMat4("u_invProjMatrix", glm::inverse(camera.projection));
    m_shader->setVec4("u_viewport", camera.viewport);
    m_shader->setVec3("viewPos", camera.view_pos);
    m_shader->setFloat("u_pointRadius", m_radius);

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

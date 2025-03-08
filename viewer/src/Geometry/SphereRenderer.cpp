#include "SphereRenderer.h"
#include <glad/glad.h>
#include <Render/Shader.h>
#include <Render/RenderObject.h>

SphereRenderer::SphereRenderer(const glm::vec2 &material, float radius) : m_material(material), m_radius(radius)
{
    shader = std::make_shared<Shader>("sphere_raycast", true);
}

void SphereRenderer::Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos, const RenderObject *object)
{
    assert(light_infos.size() <= SphereRenderer::MAX_LIGHTS);

    shader->use();
    shader->setMat4("model", object->model);
    shader->setMat4("view", camera.view);

    shader->setMat4("u_projMatrix", camera.projection);
    shader->setMat4("u_invProjMatrix", glm::inverse(camera.projection));
    shader->setVec4("u_viewport", camera.viewport);
    shader->setVec3("viewPos", camera.viewPos);
    shader->setFloat("u_pointRadius", m_radius);

    for (size_t i = 0; i < light_infos.size(); ++i)
    {
        shader->setVec3("lightPos[" + std::to_string(i) + "]", light_infos[i].pos);
        shader->setVec3("lightColor[" + std::to_string(i) + "]", light_infos[i].color);
        shader->setBool("lightsOn[" + std::to_string(i) + "]", light_infos[i].isOn);
    }
    shader->setFloat("metalicIn", m_material.x);
    shader->setFloat("roughnessIn", m_material.y);

    object->DrawVAO();
}

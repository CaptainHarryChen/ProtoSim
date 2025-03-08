#include "PbrRenderer.h"
#include <glad/glad.h>
#include <stb_image.h>
#include <Render/Shader.h>
#include <Mesh/Mesh.h>

PbrRenderer::PbrRenderer(const std::vector<glm::vec3> &material)
    : m_material(material)
{
    m_shader = std::make_shared<Shader>("pbr", true);
}

void PbrRenderer::Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos, const RenderObject *object)
{
    assert(light_infos.size() <= PbrRenderer::MAX_LIGHTS);
    bool enable_shadow = shadow_mapping_infos.size() > 0;
    if (enable_shadow)
        assert(shadow_mapping_infos.size() == light_infos.size());

    m_shader->use();
    m_shader->setMat4("model", object->m_model_mat);
    m_shader->setMat4("view", camera.view);
    m_shader->setMat4("projection", camera.projection);
    m_shader->setVec3("viewPos", camera.view_pos);
    m_shader->setBool("enableShadow", enable_shadow);

    for (size_t i = 0; i < light_infos.size(); ++i)
    {
        m_shader->setVec3("lightPos[" + std::to_string(i) + "]", light_infos[i].pos);
        m_shader->setVec3("lightColor[" + std::to_string(i) + "]", light_infos[i].color);
        m_shader->setBool("lightsOn[" + std::to_string(i) + "]", light_infos[i].is_on);
    }
    for (size_t i = light_infos.size(); i < PbrRenderer::MAX_LIGHTS; ++i)
        m_shader->setBool("lightsOn[" + std::to_string(i) + "]", false);

    m_shader->setVec3("albedoIn", m_material[0]);
    m_shader->setFloat("metallicIn", m_material[1][0]);
    m_shader->setFloat("roughnessIn", m_material[1][1]);
    m_shader->setFloat("aoIn", m_material[1][2]);
    for (size_t i = 0; i < shadow_mapping_infos.size(); ++i)
    {
        glActiveTexture(GL_TEXTURE0 + (int)i);
        glBindTexture(GL_TEXTURE_CUBE_MAP, shadow_mapping_infos[i].depth_map);
    }
    for (size_t i = 0; i < shadow_mapping_infos.size(); ++i)
    {
        m_shader->setInt("depthMap3D[" + std::to_string(i) + "]", (int)i);
        m_shader->setFloat("far_plane_of_depth_map[" + std::to_string(i) + "]", shadow_mapping_infos[i].far_plane);
    }
    object->DrawVAO();
}

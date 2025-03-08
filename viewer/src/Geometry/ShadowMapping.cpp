#include "ShadowMapping.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glad/glad.h>
#include <Render/Shader.h>
#include <Mesh/Mesh.h>

ShadowMapping::ShadowMapping(float near_plane, float far_plane, unsigned int width, unsigned int height)
    : m_near_plane(near_plane), m_far_plane(far_plane), m_shadow_width(width), m_shadow_height(height)
{
    m_depth_shader = std::make_shared<Shader>("point_shadow_mapping_depth", true);
    glGenFramebuffers(1, &m_FBO);
    glGenTextures(1, &m_depth_map);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_depth_map);
    for (unsigned int i = 0; i < 6; ++i)
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_DEPTH_COMPONENT, m_shadow_width, m_shadow_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, m_depth_map, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ShadowMapping::Draw(glm::vec3 light_pos, std::vector<std::shared_ptr<RenderObject>> &objects)
{
    glm::mat4 shadowProj = glm::perspective(glm::radians(90.0f), (float)m_shadow_width / (float)m_shadow_height, m_near_plane, m_far_plane);
    std::vector<glm::mat4> shadowTransforms;
    shadowTransforms.push_back(shadowProj * glm::lookAt(light_pos, light_pos + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)));
    shadowTransforms.push_back(shadowProj * glm::lookAt(light_pos, light_pos + glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)));
    shadowTransforms.push_back(shadowProj * glm::lookAt(light_pos, light_pos + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)));
    shadowTransforms.push_back(shadowProj * glm::lookAt(light_pos, light_pos + glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)));
    shadowTransforms.push_back(shadowProj * glm::lookAt(light_pos, light_pos + glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)));
    shadowTransforms.push_back(shadowProj * glm::lookAt(light_pos, light_pos + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f)));

    glViewport(0, 0, m_shadow_width, m_shadow_height);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glClear(GL_DEPTH_BUFFER_BIT);
    m_depth_shader->use();
    for (unsigned int i = 0; i < 6; ++i)
        m_depth_shader->setMat4("shadowMatrices[" + std::to_string(i) + "]", shadowTransforms[i]);
    m_depth_shader->setFloat("far_plane", m_far_plane);
    m_depth_shader->setVec3("lightPos", light_pos);

    glClear(GL_DEPTH_BUFFER_BIT);
    for (auto &object : objects)
    {
        auto mesh = std::dynamic_pointer_cast<Mesh>(object); // only support SurfaceMesh to generate shadow map for now
        if (mesh == nullptr)
            continue;
        m_depth_shader->setMat4("model", mesh->m_model_mat);
        mesh->DrawVAO();
    }
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

ShadowMappingInfo ShadowMapping::GetShadowMappingInfo()
{
    return ShadowMappingInfo{m_depth_map, m_far_plane};
}

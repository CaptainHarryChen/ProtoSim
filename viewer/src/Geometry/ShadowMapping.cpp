#include "ShadowMapping.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glad/glad.h>
#include <Render/Shader.h>
#include <Mesh/Mesh.h>

ShadowMapping::ShadowMapping(float near_plane, float far_plane, unsigned int width, unsigned int height)
    : near_plane(near_plane), far_plane(far_plane), shadow_width(width), shadow_height(height)
{
    depth_shader = std::make_shared<Shader>("point_shadow_mapping_depth", true);
    glGenFramebuffers(1, &FBO);
    glGenTextures(1, &depthMap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, depthMap);
    for (unsigned int i = 0; i < 6; ++i)
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_DEPTH_COMPONENT, shadow_width, shadow_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthMap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ShadowMapping::Draw(glm::vec3 lightPos, std::vector<std::shared_ptr<RenderObject>> &objects)
{
    glm::mat4 shadowProj = glm::perspective(glm::radians(90.0f), (float)shadow_width / (float)shadow_height, near_plane, far_plane);
    std::vector<glm::mat4> shadowTransforms;
    shadowTransforms.push_back(shadowProj * glm::lookAt(lightPos, lightPos + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)));
    shadowTransforms.push_back(shadowProj * glm::lookAt(lightPos, lightPos + glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)));
    shadowTransforms.push_back(shadowProj * glm::lookAt(lightPos, lightPos + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)));
    shadowTransforms.push_back(shadowProj * glm::lookAt(lightPos, lightPos + glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)));
    shadowTransforms.push_back(shadowProj * glm::lookAt(lightPos, lightPos + glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)));
    shadowTransforms.push_back(shadowProj * glm::lookAt(lightPos, lightPos + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f)));

    glViewport(0, 0, shadow_width, shadow_height);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO);
    glClear(GL_DEPTH_BUFFER_BIT);
    depth_shader->use();
    for (unsigned int i = 0; i < 6; ++i)
        depth_shader->setMat4("shadowMatrices[" + std::to_string(i) + "]", shadowTransforms[i]);
    depth_shader->setFloat("far_plane", far_plane);
    depth_shader->setVec3("lightPos", lightPos);

    glClear(GL_DEPTH_BUFFER_BIT);
    for (auto &object : objects)
    {
        auto mesh = std::dynamic_pointer_cast<Mesh>(object); // only support SurfaceMesh to generate shadow map for now
        if (mesh == nullptr)
            continue;
        depth_shader->setMat4("model", mesh->model);
        mesh->Draw();
    }
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

ShadowMappingInfo ShadowMapping::GetShadowMappingInfo()
{
    return ShadowMappingInfo{depthMap, far_plane};
}

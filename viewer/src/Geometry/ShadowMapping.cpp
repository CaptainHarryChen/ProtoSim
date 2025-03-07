#include "ShadowMapping.h"
#include <glad/glad.h>
#include <Render/Shader.h>
#include <Mesh/SurfaceMesh.h>

ShadowMapping::ShadowMapping(bool cubic)
{
    isCubic = cubic;
    if (isCubic)
    {
        shadow_width = 1024, shadow_height = 1024;
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
    else
    {
        depth_shader = std::make_shared<Shader>("shadow_mapping_depth");
        glGenFramebuffers(1, &FBO);
        // create depth texture
        glGenTextures(1, &depthMap);
        glBindTexture(GL_TEXTURE_2D, depthMap);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadow_width,
                     shadow_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        float borderColor[] = {1.0, 1.0, 1.0, 1.0};
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
        // attach depth texture as FBO's depth buffer
        glBindFramebuffer(GL_FRAMEBUFFER, FBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
                               depthMap, 0);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
}

void ShadowMapping::Draw(glm::vec3 lightPos, std::vector<std::shared_ptr<SurfaceMesh>> &meshes)
{
    if (isCubic)
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
        depth_shader->setMat4("model", glm::mat4(1.0f));

        glClear(GL_DEPTH_BUFFER_BIT);
        for (int i = 0; i < meshes.size(); ++i)
        {
            meshes[i]->Draw();
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    else
    {
        glm::mat4 lightProjection, lightView;
        lightProjection = glm::perspective(glm::radians(150.0f), (GLfloat)shadow_width / (GLfloat)shadow_height, near_plane, far_plane);
        lightView = glm::lookAt(lightPos, glm::vec3(0.0f), glm::vec3(0.0, 1.0, 0.0));

        lightSpaceMatrix = lightProjection * lightView;
        depth_shader->use();
        depth_shader->setMat4("lightSpaceMatrix", lightSpaceMatrix);
        depth_shader->setMat4("model", glm::mat4(1.0));
        glViewport(0, 0, shadow_width, shadow_height);
        glBindFramebuffer(GL_FRAMEBUFFER, FBO);
        glClear(GL_DEPTH_BUFFER_BIT);

        for (int i = 0; i < meshes.size(); ++i)
        {
            meshes[i]->Draw();
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
}

unsigned int ShadowMapping::getDepthMap()
{
    return depthMap;
}

glm::mat4 ShadowMapping::getlightSpaceMatrix()
{
    return lightSpaceMatrix;
}

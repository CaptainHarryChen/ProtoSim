#include "TextureRenderer.h"
#include <glad/glad.h>
#include <stb_image.h>
#include <viewer/Framework/Shader.h>
#include <viewer/RenderObject/Mesh.h>
#include <viewer/Framework/Camera.h>

namespace viewer {

TextureRenderer::TextureRenderer(std::vector<std::string> textures)
{
    assert(textures.size() == 5);
    m_albedo_id    = LoadTexture(textures[0].c_str());
    m_normal_id    = LoadTexture(textures[1].c_str());
    m_metallic_id  = LoadTexture(textures[2].c_str());
    m_roughness_id = LoadTexture(textures[3].c_str());
    m_ao_id        = LoadTexture(textures[4].c_str());
    m_shader       = std::make_shared<Shader>("pbr_texture", true);
}

void TextureRenderer::Draw(const CameraInfo& camera, const std::vector<LightInfo>& light_infos, const std::vector<ShadowMappingInfo>& shadow_mapping_infos, const RenderObject* object) const
{
    assert(light_infos.size() <= TextureRenderer::MAX_LIGHTS);
    bool enable_shadow = shadow_mapping_infos.size() > 0;
    if (enable_shadow)
        assert(shadow_mapping_infos.size() == light_infos.size());

    m_shader->use();
    m_shader->setMat4("model", object->GetModelMatrix());
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
    // set the rest lights off
    for (size_t i = light_infos.size(); i < TextureRenderer::MAX_LIGHTS; ++i)
        m_shader->setBool("lightsOn[" + std::to_string(i) + "]", false);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_albedo_id);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_normal_id);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, m_metallic_id);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, m_roughness_id);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, m_ao_id);
    for (size_t i = 0; i < shadow_mapping_infos.size(); ++i)
    {
        glActiveTexture(GL_TEXTURE5 + ( int )i);
        glBindTexture(GL_TEXTURE_CUBE_MAP, shadow_mapping_infos[i].depth_map);
    }
    m_shader->setInt("albedoMap", 0);
    m_shader->setInt("normalMap", 1);
    m_shader->setInt("metallicMap", 2);
    m_shader->setInt("roughnessMap", 3);
    m_shader->setInt("aoMap", 4);
    for (size_t i = 0; i < shadow_mapping_infos.size(); ++i)
    {
        m_shader->setInt("depthMap3D[" + std::to_string(i) + "]", ( int )i + 5);
        m_shader->setFloat("far_plane_of_depth_map[" + std::to_string(i) + "]", shadow_mapping_infos[i].far_plane);
    }
    // set the rest shadow maps to the first shadow map, preventing texture conflict with albedo
    for (size_t i = shadow_mapping_infos.size(); i < TextureRenderer::MAX_LIGHTS; ++i)
    {
        m_shader->setInt("depthMap3D[" + std::to_string(i) + "]", 5);
        m_shader->setFloat("far_plane_of_depth_map[" + std::to_string(i) + "]", 0.0f);
    }
    object->DrawVAO();
}

unsigned int TextureRenderer::LoadTexture(const char* path) const
{
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    stbi_set_flip_vertically_on_load(true);

    int            texture_width, texture_height, nrChannels;
    unsigned char* texture_data = stbi_load(path, &texture_width, &texture_height, &nrChannels, 0);
    GLenum         format;
    if (nrChannels == 1)
        format = GL_RED;
    if (nrChannels == 3)
        format = GL_RGB;
    if (nrChannels == 4)
        format = GL_RGBA;

    glTexImage2D(GL_TEXTURE_2D, 0, format, texture_width, texture_height, 0, format, GL_UNSIGNED_BYTE, texture_data);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(texture_data);

    return texture;
}

}  // namespace viewer

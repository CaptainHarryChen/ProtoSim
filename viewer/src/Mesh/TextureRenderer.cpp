#include "TextureRenderer.h"
#include <glad/glad.h>
#include <stb_image.h>
#include <Render/Shader.h>
#include <Mesh/Mesh.h>

TextureRenderer::TextureRenderer(std::vector<std::string> textures)
{
    assert(textures.size() == 5);
    albedoID = loadTexture(textures[0].c_str());
    normalID = loadTexture(textures[1].c_str());
    metallicID = loadTexture(textures[2].c_str());
    roughnessID = loadTexture(textures[3].c_str());
    aoID = loadTexture(textures[4].c_str());
    shader = std::make_shared<Shader>("pbr_texture", true);
}

void TextureRenderer::Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos, const RenderObject *object)
{
    assert(light_infos.size() <= TextureRenderer::MAX_LIGHTS);
    bool enable_shadow = shadow_mapping_infos.size() > 0;
    if (enable_shadow)
        assert(shadow_mapping_infos.size() == light_infos.size());

    shader->use();
    shader->setMat4("model", object->model);
    shader->setMat4("view", camera.view);
    shader->setMat4("projection", camera.projection);
    shader->setVec3("viewPos", camera.viewPos);
    shader->setBool("enableShadow", enable_shadow);

    for (size_t i = 0; i < light_infos.size(); ++i)
    {
        shader->setVec3("lightPos[" + std::to_string(i) + "]", light_infos[i].pos);
        shader->setVec3("lightColor[" + std::to_string(i) + "]", light_infos[i].color);
        shader->setBool("lightsOn[" + std::to_string(i) + "]", light_infos[i].isOn);
    }
    // set the rest lights off
    for (size_t i = light_infos.size(); i < TextureRenderer::MAX_LIGHTS; ++i)
        shader->setBool("lightsOn[" + std::to_string(i) + "]", false);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, albedoID);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, normalID);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, metallicID);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, roughnessID);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, aoID);
    for (size_t i = 0; i < shadow_mapping_infos.size(); ++i)
    {
        glActiveTexture(GL_TEXTURE5 + (int)i);
        glBindTexture(GL_TEXTURE_CUBE_MAP, shadow_mapping_infos[i].depth_map);
    }
    shader->setInt("albedoMap", 0);
    shader->setInt("normalMap", 1);
    shader->setInt("metallicMap", 2);
    shader->setInt("roughnessMap", 3);
    shader->setInt("aoMap", 4);
    for (size_t i = 0; i < shadow_mapping_infos.size(); ++i)
    {
        shader->setInt("depthMap3D[" + std::to_string(i) + "]", (int)i + 5);
        shader->setFloat("far_plane_of_depth_map[" + std::to_string(i) + "]", shadow_mapping_infos[i].far_plane);
    }
    // set the rest shadow maps to the first shadow map, preventing texture conflict with albedo
    for (size_t i = shadow_mapping_infos.size(); i < TextureRenderer::MAX_LIGHTS; ++i)
    {
        shader->setInt("depthMap3D[" + std::to_string(i) + "]", 5);
        shader->setFloat("far_plane_of_depth_map[" + std::to_string(i) + "]", 0.0f);
    }
    object->DrawVAO();
}

unsigned int TextureRenderer::loadTexture(const char *path)
{
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    stbi_set_flip_vertically_on_load(true);

    int texture_width, texture_height, nrChannels;
    unsigned char *texture_data = stbi_load(path, &texture_width, &texture_height, &nrChannels, 0);
    GLenum format;
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

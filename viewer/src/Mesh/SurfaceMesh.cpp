#include "SurfaceMesh.h"
#include <Render/Shader.h>
#include <stb_image.h>

SurfaceMesh::SurfaceMesh(const std::vector<Vertex> &vertices, const std::vector<unsigned int> &indices, const std::vector<glm::vec3> &material)
    : Mesh(vertices, indices), material(material)
{
    shader = std::make_shared<Shader>("pbr", true);
}

void SurfaceMesh::Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos)
{
    Draw(camera, light_infos, std::vector<ShadowMappingInfo>());
}

void SurfaceMesh::Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos)
{
    assert(light_infos.size() < SurfaceMesh::MAX_LIGHTS);
    bool enable_shadow = shadow_mapping_infos.size() > 0;
    if (enable_shadow)
        assert(shadow_mapping_infos.size() == light_infos.size());

    shader->use();
    shader->setMat4("model", this->model);
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
    for (size_t i = light_infos.size(); i < SurfaceMesh::MAX_LIGHTS; ++i)
        shader->setBool("lightsOn[" + std::to_string(i) + "]", false);

    shader->setVec3("albedoIn", material[0]);
    shader->setFloat("metallicIn", material[1][0]);
    shader->setFloat("roughnessIn", material[1][1]);
    shader->setFloat("aoIn", material[1][2]);
    for (size_t i = 0; i < shadow_mapping_infos.size(); ++i)
    {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_CUBE_MAP, shadow_mapping_infos[i].depth_map);
    }
    for (size_t i = 0; i < shadow_mapping_infos.size(); ++i)
    {
        shader->setInt("depthMap3D[" + std::to_string(i) + "]", i);
        shader->setFloat("far_plane_of_depth_map[" + std::to_string(i) + "]", shadow_mapping_infos[i].far_plane);
    }
    Mesh::Draw();
}

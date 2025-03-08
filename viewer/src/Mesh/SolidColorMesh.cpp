#include "SolidColorMesh.h"
#include <Render/Shader.h>

SolidColorMesh::SolidColorMesh(const std::vector<Vertex> &vertices, const std::vector<unsigned int> &indices, const glm::vec3 &color)
    : Mesh(vertices, indices), color(color)
{
    shader = std::make_shared<Shader>("solid_color", false);
}

void SolidColorMesh::Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos)
{
    shader->use();
    shader->setMat4("model", this->model);
    shader->setMat4("view", camera.view);
    shader->setMat4("projection", camera.projection);
    shader->setVec3("color", color);
    Mesh::Draw();
}

void SolidColorMesh::Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos)
{
    Draw(camera, light_infos);
}

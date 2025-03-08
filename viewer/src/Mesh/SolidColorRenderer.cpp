#include "SolidColorRenderer.h"
#include <Render/Shader.h>
#include <Mesh/Mesh.h>

SolidColorRenderer::SolidColorRenderer(const glm::vec3 &color)
    : color(color)
{
    shader = std::make_shared<Shader>("solid_color", false);
}

void SolidColorRenderer::Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos, const RenderObject *object)
{
    auto mesh = dynamic_cast<const Mesh *>(object);
    assert(mesh != nullptr);

    shader->use();
    shader->setMat4("model", mesh->model);
    shader->setMat4("view", camera.view);
    shader->setMat4("projection", camera.projection);
    shader->setVec3("color", color);
    mesh->DrawVAO();
}

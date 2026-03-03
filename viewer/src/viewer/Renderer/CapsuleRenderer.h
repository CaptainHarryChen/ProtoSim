#pragma once

#include <glm/glm.hpp>
#include <viewer/Framework/Renderer.h>

namespace viewer
{

class CapsuleRenderer : public Renderer
{
    static const int MAX_LIGHTS = 4;

public:
    CapsuleRenderer(const glm::vec2 &material, float radius, float half_height);
    virtual ~CapsuleRenderer() = default;

    virtual void Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos,
                      const std::vector<ShadowMappingInfo> &shadow_mapping_infos,
                      const RenderObject *object) const override;

    void SetMaterial(const glm::vec2 &material);

protected:
    glm::vec2 m_material;
    float m_radius;
    float m_half_height;
};

}

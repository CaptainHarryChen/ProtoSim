#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <viewer/Framework/Renderer.h>

namespace viewer {

class SphereRenderer : public Renderer
{
    static const int MAX_LIGHTS = 4;
public:
    SphereRenderer(const glm::vec2& material, float radius, bool highlightRolling = false);
    virtual ~SphereRenderer() = default;

    virtual void Draw(const CameraInfo& camera, const std::vector<LightInfo>& light_infos, const std::vector<ShadowMappingInfo>& shadow_mapping_infos, const RenderObject* object) const override;

protected:
    glm::vec2 m_material;
    float     m_radius;
    bool      m_highlightRolling = false;
};

}  // namespace viewer

#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <Render/Renderer.h>

class SphereRenderer : public Renderer
{
    static const int MAX_LIGHTS = 4; // limited by the shader
public:
    /// @brief Construct a new Sphere object with a material and radius
    /// @param material vec2(x, y), x is the metalic, y is the roughness
    SphereRenderer(const glm::vec2 &material, float radius);
	virtual ~SphereRenderer() = default;

	virtual void Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos, const RenderObject *object) const override;

	void SetMaterial(const glm::vec2 &material);

protected:
	glm::vec2 m_material;
    float m_radius;
};

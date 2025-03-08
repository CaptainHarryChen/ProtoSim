#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <Render/Renderer.h>

class PbrRenderer : public Renderer
{
	static const int MAX_LIGHTS = 4; // limited by the shader

public:
	/// @brief Construct a new Mesh object with pbr material
	/// @param material an array of 2 vec3, representing the albedo and <metallic, roughness, ao>
	PbrRenderer(const std::vector<glm::vec3> &material = {glm::vec3(1.0f), glm::vec3(0.1f, 0.1f, 0.1f)});

	virtual void Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos, const RenderObject *object) override;

protected:
	std::vector<glm::vec3> material; // pbr [<albedo>, <metallic, roughness, ao>]
};

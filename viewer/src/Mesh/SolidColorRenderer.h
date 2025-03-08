#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <Render/Renderer.h>

class SolidColorRenderer : public Renderer
{
public:
	SolidColorRenderer(const glm::vec3 &color = glm::vec3(1.0f), bool only_edge = false);
	virtual ~SolidColorRenderer() = default;
	virtual void Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos, const RenderObject *object) override;
	void SetColor(const glm::vec3 &color);

protected:
	bool m_only_edge;
	glm::vec3 color;
};

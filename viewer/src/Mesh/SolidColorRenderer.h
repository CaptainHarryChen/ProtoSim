#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <cassert>
#include <string>
#include <vector>
#include <Render/Renderer.h>

class SolidColorRenderer : public Renderer
{
public:
	SolidColorRenderer(const glm::vec3 &color = glm::vec3(1.0f));
	virtual ~SolidColorRenderer() = default;
	virtual void Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos, const RenderObject *object) override;

protected:
	glm::vec3 color;
};

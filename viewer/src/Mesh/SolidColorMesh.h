#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <cassert>
#include <string>
#include <vector>
#include <Mesh/Mesh.h>

class Shader;
class ShadowMapping;

class SolidColorMesh : public Mesh
{
public:
	SolidColorMesh(const std::vector<Vertex> &vertices, const std::vector<unsigned int> &indices, const glm::vec3 &color = glm::vec3(1.0f));
	virtual ~SolidColorMesh() = default; 

	virtual void Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos) override;
	virtual void Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos) override;

protected:
	std::shared_ptr<Shader> shader;

    glm::vec3 color;
};

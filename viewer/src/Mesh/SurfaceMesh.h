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

class SurfaceMesh : public Mesh
{
	static const int MAX_LIGHTS = 4; // limited by the shader

public:
	/// @brief Construct a new Mesh object with pbr material
	/// @param vertices 
	/// @param indices 
	/// @param material an array of 2 vec3, representing the albedo and <metallic, roughness, ao>
	SurfaceMesh(const std::vector<Vertex> &vertices, const std::vector<unsigned int> &indices, const std::vector<glm::vec3> &material = {glm::vec3(1.0f), glm::vec3(0.1f, 0.1f, 0.1f)});

	std::vector<glm::vec3> material; // pbr [<albedo>, <metallic, roughness, ao>]

	virtual void Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos) override;
	virtual void Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos) override;

protected:
	std::shared_ptr<Shader> shader;
};

#pragma once
#include <string>
#include <glm/gtx/quaternion.hpp>
#include <Mesh/MeshBase.h>
#include <viewer_config.h>

class MeshLoader: public MeshBase
{
public:
	MeshLoader(std::string inputfile, float scale,
		glm::vec3 translate,
		glm::vec3 rotate, // pitch, yaw, roll
		std::vector<glm::vec3> material,
		bool isCubic);

	MeshLoader(std::string inputfile, float scale,
		glm::vec3 translate,
		glm::vec3 rotate, // pitch, yaw, roll
		bool isCubic,
		std::string albedo = std::string(VIEWER_DIR) + "/data/default_texture/albedo.png",
		std::string metallic = std::string(VIEWER_DIR) + "/data/default_texture/metallic.png",
		std::string normal = std::string(VIEWER_DIR) + "/data/default_texture/normal.png",
		std::string roughness = std::string(VIEWER_DIR) + "/data/default_texture/roughness.png",
		std::string ao = std::string(VIEWER_DIR) + "/data/default_texture/ao.png");
    
	std::vector<Vertex> vertices;
	std::vector<unsigned int> indices;

private:
	std::string path;

	glm::vec3 rotateVecQuat(glm::quat q, glm::vec3 v);
};

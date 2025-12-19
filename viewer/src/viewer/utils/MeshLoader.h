#pragma once
#include <string>
#include <memory>
#include <vector>
#include <glm/glm.hpp>
#include <viewer/viewer_config.h>

namespace viewer {

class Mesh;

namespace MeshLoader {
std::shared_ptr<Mesh> LoadMesh(std::string inputfile, float scale, glm::vec3 translate,
                               glm::vec3              rotate,  // pitch, yaw, roll
                               std::vector<glm::vec3> material);

std::shared_ptr<Mesh> LoadMesh(std::string inputfile, float scale, glm::vec3 translate,
                               glm::vec3   rotate,  // pitch, yaw, roll
                               std::string albedo    = std::string(VIEWER_DIR) + "/data/default_texture/albedo.png",
                               std::string metallic  = std::string(VIEWER_DIR) + "/data/default_texture/metallic.png",
                               std::string normal    = std::string(VIEWER_DIR) + "/data/default_texture/normal.png",
                               std::string roughness = std::string(VIEWER_DIR) + "/data/default_texture/roughness.png",
                               std::string ao        = std::string(VIEWER_DIR) + "/data/default_texture/ao.png");
};  // namespace MeshLoader

}  // namespace viewer

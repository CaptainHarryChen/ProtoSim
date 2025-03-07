#pragma once
#include <memory>
#include <vector>
#include <glm/glm.hpp>

class Shader;
class SurfaceMesh;

class ShadowMapping
{
public:
    ShadowMapping(bool cubic = true);

    void Draw(glm::vec3 lightPos, std::vector<std::shared_ptr<SurfaceMesh>> &meshes);
    unsigned int getDepthMap();
    glm::mat4 getlightSpaceMatrix();

    float near_plane = 0.1f, far_plane = 1000.f;

protected:
    std::shared_ptr<Shader> depth_shader;
    unsigned int shadow_width = 4096, shadow_height = 4096;
    unsigned int FBO;
    unsigned int depthMap;
    glm::mat4 lightSpaceMatrix;
    bool isCubic = false;
};

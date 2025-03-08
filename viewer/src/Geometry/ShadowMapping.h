#pragma once
#include <memory>
#include <vector>
#include <glm/glm.hpp>

class Shader;
class RenderObject;

struct ShadowMappingInfo
{
    unsigned int depth_map;
    float far_plane; // texture value multiply far_plane to get the real distance
};

class ShadowMapping
{
public:
    ShadowMapping(float near_plane = 0.2f, float far_plane = 1000.f, unsigned int width = 1024u, unsigned int height = 1024u);

    void Draw(glm::vec3 lightPos, std::vector<std::shared_ptr<RenderObject>> &objects);
    ShadowMappingInfo GetShadowMappingInfo();

protected:
    std::shared_ptr<Shader> depth_shader;
    float near_plane, far_plane;
    unsigned int shadow_width, shadow_height;

    unsigned int FBO;
    unsigned int depthMap;
};

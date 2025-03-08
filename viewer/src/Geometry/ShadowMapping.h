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

    void Draw(glm::vec3 light_pos, std::vector<std::shared_ptr<RenderObject>> &objects) const;
    ShadowMappingInfo GetShadowMappingInfo() const;

protected:
    std::shared_ptr<Shader> m_depth_shader;
    float m_near_plane, m_far_plane;
    unsigned int m_shadow_width, m_shadow_height;

    unsigned int m_FBO;
    unsigned int m_depth_map;
};

#pragma once
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <Render/Light.h>
#include <Geometry/ShadowMapping.h>

struct CameraInfo
{
    glm::mat4 view;
    glm::mat4 projection;
    glm::vec3 viewPos;
};

class Renderer;

class RenderObject
{
public:
    RenderObject() = default;
    virtual ~RenderObject() = default;

    /// @brief the model transform of the object
    /// @details for mesh object, this is usually the identity matrix. Because the vertices are usually already in the world space.
    glm::mat4 model = glm::mat4(1.0f);
    bool m_enable = true;

    virtual void AddRenderer(std::shared_ptr<Renderer> renderer);
    virtual void Draw(const CameraInfo &camera_info, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos = {});

    std::vector<std::shared_ptr<Renderer>> renderers;
};

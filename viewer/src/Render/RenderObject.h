#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <Render/Light.h>
#include <Geometry/ShadowMapping.h>

struct CameraInfo
{
    glm::mat4 view;
    glm::mat4 projection;
    glm::vec3 viewPos;
};

class RenderObject
{
public:
    RenderObject() = default;
    virtual ~RenderObject() = default;

    /// @brief the model transform of the object
    /// @details for mesh object, this is usually the identity matrix. Because the vertices are usually already in the world space.
    glm::mat4 model = glm::mat4(1.0f);

    virtual void Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos);
    virtual void Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos);
};

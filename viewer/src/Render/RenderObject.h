#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <Render/Light.h>
#include <Geometry/ShadowMapping.h>

struct CameraInfo
{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 projection;
    glm::vec3 viewPos;
};

class RenderObject
{
public:
    RenderObject() = default;
    virtual ~RenderObject() = default;

    virtual void Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos);
    virtual void Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos);
};

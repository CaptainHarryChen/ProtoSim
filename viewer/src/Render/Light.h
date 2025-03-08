#pragma once
#include <vector>
#include <memory>
#include <glm/glm.hpp>

struct LightInfo
{
    glm::vec3 pos;
    glm::vec3 color;
    bool isOn;
};

class RenderObject;
class ShadowMapping;
struct ShadowMappingInfo;

class Light
{
public:
    Light(glm::vec3 position, glm::vec3 color = {1.0f, 1.0f, 1.0f}, float shadow_near_plane = 0.2f, float shadow_far_plane = 1000.f);
    virtual ~Light() = default;

    virtual LightInfo GetLightInfo();
    virtual ShadowMappingInfo CreateShadowMappingInfo(std::vector<std::shared_ptr<RenderObject>> &objects);

protected:
    glm::vec3 lightPos;
    glm::vec3 lightColor;
    bool isOn = true;

    std::shared_ptr<ShadowMapping> shadowMapping;
};

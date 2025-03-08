#include "Light.h"
#include <Geometry/ShadowMapping.h>

Light::Light(glm::vec3 position, glm::vec3 color, float shadow_near_plane, float shadow_far_plane) : lightPos(position), lightColor(color)
{
    shadowMapping = std::make_shared<ShadowMapping>(shadow_near_plane, shadow_far_plane);
}

LightInfo Light::GetLightInfo()
{
    return LightInfo{lightPos, lightColor, isOn};
}

ShadowMappingInfo Light::CreateShadowMappingInfo(std::vector<std::shared_ptr<RenderObject>> &objects)
{
    shadowMapping->Draw(lightPos, objects);
    return shadowMapping->GetShadowMappingInfo();
}

#include "Light.h"
#include <Geometry/ShadowMapping.h>

Light::Light(glm::vec3 position, glm::vec3 color, float shadow_near_plane, float shadow_far_plane) : m_light_pos(position), m_light_color(color)
{
    m_shadow_mapping = std::make_shared<ShadowMapping>(shadow_near_plane, shadow_far_plane);
}

LightInfo Light::GetLightInfo() const
{
    return LightInfo{m_light_pos, m_light_color, m_is_on};
}

ShadowMappingInfo Light::CreateShadowMappingInfo(std::vector<std::shared_ptr<RenderObject>> &objects) const
{
    m_shadow_mapping->Draw(m_light_pos, objects);
    return m_shadow_mapping->GetShadowMappingInfo();
}

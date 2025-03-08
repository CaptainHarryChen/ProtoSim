#include "RenderObject.h"
#include <Render/Renderer.h>

void RenderObject::AddRenderer(std::shared_ptr<Renderer> renderer)
{
    m_renderers.push_back(renderer);
}

void RenderObject::Draw(const CameraInfo &camera_info, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos)
{
    if (!m_enable)
        return;
    for (auto &renderer : m_renderers)
        renderer->Draw(camera_info, light_infos, shadow_mapping_infos, this);
}

void RenderObject::DrawVAO() const
{
    // Do nothing
}

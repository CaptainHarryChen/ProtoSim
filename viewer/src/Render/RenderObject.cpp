#include "RenderObject.h"
#include <Render/Renderer.h>

void RenderObject::AddRenderer(std::shared_ptr<Renderer> renderer)
{
    renderers.push_back(renderer);
}

void RenderObject::Draw(const CameraInfo &camera_info, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos)
{
    for (auto &renderer : renderers)
        renderer->Draw(camera_info, light_infos, shadow_mapping_infos, this);
}

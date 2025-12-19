#pragma once
#include <vector>
#include <memory>

namespace viewer {

struct CameraInfo;
struct LightInfo;
struct ShadowMappingInfo;
class Shader;
class RenderObject;

class Renderer
{
public:
    Renderer()          = default;
    virtual ~Renderer() = default;

    virtual void Draw(const CameraInfo& camera_info, const std::vector<LightInfo>& light_infos, const std::vector<ShadowMappingInfo>& shadow_mapping_infos, const RenderObject* object) const = 0;

protected:
    std::shared_ptr<Shader> m_shader;
};

}  // namespace viewer

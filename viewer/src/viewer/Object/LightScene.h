#pragma once
#include <vector>
#include <memory>
#include <viewer/Framework/RenderObject.h>
#include <viewer/Framework/Object.h>

namespace viewer {

class LightSceneControlGUI : public RenderObject
{
public:
    LightSceneControlGUI()          = default;
    virtual ~LightSceneControlGUI() = default;

    virtual void Draw(const CameraInfo& camera_info, const std::vector<LightInfo>& light_infos, const std::vector<ShadowMappingInfo>& shadow_mapping_infos) override;

    size_t                 m_light_num   = 4;
    std::vector<char>      m_light_on    = { true, false, false, true };  // vector<bool> is optimized stupidly in C++, so use char instead
    std::vector<glm::vec3> m_light_color = { glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f) };
    std::vector<glm::vec3> m_light_pos   = { glm::vec3(-3.0f, 3.0f, -3.0f), glm::vec3(-3.0f, 3.0f, 3.0f), glm::vec3(3.0f, 3.0f, -3.0f), glm::vec3(3.0f, 3.0f, 3.0f) };
};

class CubeLight;

class LightScene : public Object
{
public:
    LightScene();
    virtual ~LightScene() = default;

    std::vector<std::shared_ptr<CubeLight>> m_lights;
    std::shared_ptr<LightSceneControlGUI>   m_control_gui;

    virtual void Update(double delta_time) override;
};

}  // namespace viewer

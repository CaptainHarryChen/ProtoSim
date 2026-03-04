#pragma once
#include <vector>
#include <memory>
#include <viewer/Framework/Light.h>
#include <viewer/Framework/ShadowMapping.h>
#include <glm/glm.hpp>

namespace viewer {

struct CameraInfo;
struct Event;
class Renderer;

class RenderObject
{
public:
    RenderObject()          = default;
    virtual ~RenderObject() = default;

    bool m_enable = true;
    glm::mat4 m_model_mat = glm::mat4(1.0f);

    virtual void AddRenderer(std::shared_ptr<Renderer> renderer);

    /// @brief Draw the object
    /// @details This function should be called by the RenderSystem. It will call the Draw function of each renderer added to the object.
    virtual void Draw(const CameraInfo& camera_info, const std::vector<LightInfo>& light_infos, const std::vector<ShadowMappingInfo>& shadow_mapping_infos = {});

    /// @brief After the renderer setup the shader and uniform variables, it should call this function to draw the object.
    /// @details This function should be implemented by the derived class.
    virtual void DrawVAO() const;

    /// @brief Get the model matrix of the object. Usually identity matrix. Only Mesh will override this function.
    /// @return the model matrix of the object
    virtual glm::mat4 GetModelMatrix() const;

    std::vector<std::shared_ptr<Renderer>> m_renderers;
};

}  // namespace viewer

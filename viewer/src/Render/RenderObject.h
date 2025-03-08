#pragma once
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <Render/Light.h>
#include <Geometry/ShadowMapping.h>

struct CameraInfo
{
    glm::mat4 view;
    glm::mat4 projection;
    glm::vec3 view_pos;
    glm::vec4 viewport;
};

class Renderer;

class RenderObject
{
public:
    RenderObject() = default;
    virtual ~RenderObject() = default;

    /// @brief the model transform of the object
    /// @details for mesh object, this is usually the identity matrix. Because the vertices are usually already in the world space.
    glm::mat4 m_model_mat = glm::mat4(1.0f);
    bool m_enable = true;

    /// @brief Add a renderer to the object
    /// @param renderer The renderer will be called while the Draw function of the object is called. The renderer should setup the shader and uniform variables, then call the DrawVAO function of the object.
    virtual void AddRenderer(std::shared_ptr<Renderer> renderer);

    /// @brief Draw the object
    /// @details This function should be called by the RenderSystem. It will call the Draw function of each renderer added to the object.
    virtual void Draw(const CameraInfo &camera_info, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos = {});

    /// @brief After the renderer setup the shader and uniform variables, it should call this function to draw the object.
    /// @details This function should be implemented by the derived class.
    virtual void DrawVAO() const;

    std::vector<std::shared_ptr<Renderer>> m_renderers;
};

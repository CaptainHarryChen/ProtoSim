#pragma once
#include <memory>
#include <vector>
#include <string>
#include <glm/glm.hpp>

class Camera;
class Light;
class RenderObject;

class RenderSystem
{
public:
    RenderSystem();
    virtual ~RenderSystem();

    // settings
    glm::vec3 m_clear_color = {1.0, 1.0, 1.0};
    int m_viewport_width = 1600;
    int m_viewport_height = 900;

    virtual void SetCamera(std::shared_ptr<Camera> camera);
    virtual void AddLight(std::shared_ptr<Light> light);
    virtual void AddRenderObject(std::shared_ptr<RenderObject> render_object);

    virtual void RenderOneFrame();

protected:
    std::shared_ptr<Camera> m_camera;
    std::vector<std::shared_ptr<Light>> m_lights;
    std::vector<std::shared_ptr<RenderObject>> m_render_objects;
};

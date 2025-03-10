#pragma once
#include <memory>
#include <vector>
#include <Render/RenderSystem.h>
#include <Render/OrbitControl.h>

class Light;
class RenderObject;

class RenderSystem
{
public:
    static std::shared_ptr<RenderSystem> GetInstance();

    RenderSystem(std::string name = "Viewer", int init_width = 1600, int init_height = 900);
    virtual ~RenderSystem();

    // settings
    glm::vec3 m_clear_color = {0.5, 0.5, 1.0};

    virtual void AddLight(std::shared_ptr<Light> light);
    virtual void AddRenderObject(std::shared_ptr<RenderObject> render_object);

    virtual bool ProcessControl();
    virtual void RenderOneFrame();

protected:
    GLFWwindow *m_window;
    std::shared_ptr<OrbitControl> m_camera;
    std::vector<std::shared_ptr<Light>> m_lights;
    std::vector<std::shared_ptr<RenderObject>> m_render_objects;
};

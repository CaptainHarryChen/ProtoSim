#pragma once
#include <vector>
#include <memory>

class Light;
class RenderObject;

// TODO: Split the control and rendering. Split the camera and the renderer.
class RenderSystem
{
public:
    RenderSystem() = default;
    virtual ~RenderSystem() = default;
    
    virtual bool ProcessControl();
    virtual void RenderOneFrame();

    virtual void AddLight(std::shared_ptr<Light> light);
    virtual void AddRenderObject(std::shared_ptr<RenderObject> object);

protected:
    std::vector<std::shared_ptr<Light>> lights;
    std::vector<std::shared_ptr<RenderObject>> render_objects;
};

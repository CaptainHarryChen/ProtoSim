#pragma once
#include <vector>
#include <memory>

class RenderObject;

// TODO: Split the control and rendering. Split the camera and the renderer.
class RenderSystem
{
public:
    RenderSystem() = default;
    virtual ~RenderSystem() = default;
    
    virtual bool ProcessControl();
    virtual void RenderOneFrame();

    virtual void AddRenderObject(std::shared_ptr<RenderObject> object);
protected:
    std::vector<std::shared_ptr<RenderObject>> render_objects;
};

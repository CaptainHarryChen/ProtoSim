#pragma once

// TODO: Split the control and rendering. Split the camera and the renderer.
class RenderSystem
{
public:
    RenderSystem() = default;
    virtual ~RenderSystem() = default;
    
    virtual bool ProcessControl() = 0;
    virtual void RenderOneFrame() = 0;
};

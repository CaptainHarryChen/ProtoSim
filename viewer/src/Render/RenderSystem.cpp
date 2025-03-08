#include "RenderSystem.h"

bool RenderSystem::ProcessControl()
{
    return true;
}

void RenderSystem::RenderOneFrame()
{
    // do nothing
}

void RenderSystem::AddLight(std::shared_ptr<Light> light)
{
    lights.push_back(light);
}

void RenderSystem::AddRenderObject(std::shared_ptr<RenderObject> object)
{
    render_objects.push_back(object);
}

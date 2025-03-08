#include "RenderSystem.h"

bool RenderSystem::ProcessControl()
{
    return true;
}

void RenderSystem::RenderOneFrame()
{
    // do nothing
}

void RenderSystem::AddRenderObject(std::shared_ptr<RenderObject> object)
{
    render_objects.push_back(object);
}

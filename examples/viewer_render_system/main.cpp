#include <string>
#include <memory>
#include <Render/OrbitCameraRenderer.h>

int main()
{
	std::shared_ptr<RenderSystem> renderer = OrbitCameraRenderer::GetInstance();
    while(true)
    {
        if (!renderer->ProcessControl())
            break;
        renderer->RenderOneFrame();
    }
    
	return 0;
}

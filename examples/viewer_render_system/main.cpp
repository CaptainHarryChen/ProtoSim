#include <string>
#include <memory>
#include <Render/OrbitCameraRenderer.h>
#include <Object/Floor.h>
#include <Object/CubeLight.h>
#include <Mesh/MeshLoader.h>

int main()
{
    std::shared_ptr<RenderSystem> renderer = OrbitCameraRenderer::GetInstance();

    auto floor = std::make_shared<Floor>(100.0f);
    renderer->AddRenderObject(floor->getMesh());
    auto bunny1 = std::make_shared<MeshLoader>(std::string(VIEWER_DIR) + "/data/objs/venus.obj", 0.01f, glm::vec3(-1.0, 0.0, 0.0), glm::vec3(glm::radians(-90.0f), 0., 0.), std::vector<glm::vec3>({glm::vec3(1.), glm::vec3(0.1f)}));
    renderer->AddRenderObject(bunny1->getMesh());
    auto light = std::make_shared<CubeLight>(glm::vec3(-3.0f, 3.0f, -3.0f), glm::vec3(0.5, 1.0, 0.5));
    renderer->AddLight(light->getLight());
    renderer->AddRenderObject(light->getMesh());

    while (true)
    {
        if (!renderer->ProcessControl())
            break;
        renderer->RenderOneFrame();
    }

    return 0;
}

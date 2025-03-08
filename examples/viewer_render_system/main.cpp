#include <string>
#include <memory>
#include <Render/OrbitCameraRenderer.h>
#include <Object/Floor.h>
#include <Object/CubeLight.h>
#include <Mesh/MeshLoader.h>
#include <Mesh/Mesh.h>

int main()
{
    std::shared_ptr<RenderSystem> render_system = OrbitCameraRenderer::GetInstance();

    auto floor = std::make_shared<Floor>(100.0f);
    render_system->AddRenderObject(floor->GetMesh());
    auto venus = MeshLoader::LoadMesh(std::string(VIEWER_DIR) + "/data/objs/venus.obj", 0.01f, glm::vec3(-1.0, 0.0, 0.0), glm::vec3(glm::radians(-90.0f), 0., 0.), std::vector<glm::vec3>({glm::vec3(1.), glm::vec3(0.1f)}));
    render_system->AddRenderObject(venus);
    auto light = std::make_shared<CubeLight>(glm::vec3(-3.0f, 3.0f, -3.0f), glm::vec3(0.5, 1.0, 0.5));
    render_system->AddLight(light->GetLight());
    render_system->AddRenderObject(light->GetMesh());

    while (true)
    {
        if (!render_system->ProcessControl())
            break;
        render_system->RenderOneFrame();
    }

    return 0;
}

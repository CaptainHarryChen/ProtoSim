#include <string>
#include <memory>
#include <Render/OrbitCameraRenderer.h>
#include <Object/Floor.h>
#include <Object/LightScene.h>
#include <Mesh/SolidColorRenderer.h>
#include <Mesh/MeshLoader.h>
#include <Mesh/Mesh.h>
#include <Geometry/LineSegment.h>
#include <Geometry/ParticleBatch.h>
#include <Geometry/SphereRenderer.h>

int main()
{
    std::shared_ptr<RenderSystem> render_system = OrbitCameraRenderer::GetInstance();

    auto floor = std::make_shared<Floor>(100.0f);
    render_system->AddRenderObject(floor->GetMesh());

    auto venus = MeshLoader::LoadMesh(std::string(VIEWER_DIR) + "/data/objs/venus.obj", 0.01f, glm::vec3(-1.0, 0.0, 0.0), glm::vec3(glm::radians(-90.0f), 0., 0.), std::vector<glm::vec3>({glm::vec3(1.), glm::vec3(0.1f)}));
    venus->AddRenderer(std::make_shared<SolidColorRenderer>(glm::vec3(1.0, 0.0, 0.0), true)); // Edge renderer
    render_system->AddRenderObject(venus);

    auto squirel = MeshLoader::LoadMesh(std::string(VIEWER_DIR) + std::string("/data/objs/CartoonSquirelModel/CartoonSquirelModel.obj"), 0.5, glm::vec3(0., 0., -1.), glm::vec3(0.),
                                        std::string(VIEWER_DIR) + std::string("/data/objs/CartoonSquirelModel/DiffSqurel.tga"));
    render_system->AddRenderObject(squirel);

    auto dragon = MeshLoader::LoadMesh(std::string(VIEWER_DIR) + "/data/objs/dragon.obj", 6.0, glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.), {glm::vec3(0., 0., 1.), glm::vec3(0.1, 0.1, 0.1)});
    render_system->AddRenderObject(dragon);

    auto light_scene = std::make_shared<LightScene>();

    auto lines = std::make_shared<LineSegment>(std::vector<glm::vec3>({glm::vec3(-1., 0.0, -1.), glm::vec3(-1., 0.0, 1.), glm::vec3(1., 0.0, -1.), glm::vec3(1., 0.0, 1.), glm::vec3(-1., 2.0, -1.), glm::vec3(-1., 2.0, 1.), glm::vec3(1., 2.0, -1.), glm::vec3(1., 2.0, 1.)}), std::vector<std::pair<int, int>>({{0, 4}, {1, 5}, {2, 6}, {3, 7}, {0, 1}, {0, 2}, {1, 3}, {2, 3}, {4, 5}, {4, 6}, {5, 7}, {6, 7}}));
    lines->AddRenderer(std::make_shared<SolidColorRenderer>(glm::vec3(1.0, 1.0, 1.0), true));
    render_system->AddRenderObject(lines);

    auto particles = std::make_shared<ParticleBatch>(std::vector<Particle>({{glm::vec3(0.0, 4.0, 0.0), glm::vec3(1.0, 0.0, 0.0)}, {glm::vec3(0.0, 5.0, 0.0), glm::vec3(0.0, 1.0, 0.0)}, {glm::vec3(0.0, 6.0, 0.0), glm::vec3(0.0, 0.0, 1.0)}}));
    particles->AddRenderer(std::make_shared<SphereRenderer>(glm::vec2(0.1), 0.2));
    render_system->AddRenderObject(particles);

    while (true)
    {
        if (!render_system->ProcessControl())
            break;

        light_scene->Update();

        render_system->RenderOneFrame();
    }

    return 0;
}

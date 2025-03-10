#include <string>
#include <memory>
#include <GLFWApp.h>
#include <Render/RenderSystem.h>
#include <Camera/OrbitCamera.h>
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
    auto app = GLFWApp::GetInstance("App", 1600, 900);
    std::shared_ptr<RenderSystem> render_system = app->GetRenderSystem();
    
    auto camera = std::make_shared<OrbitCamera>(glm::radians(-45.0f), glm::radians(-45.0f), 10.0f);
    app->SetCamera(camera);
    app->AddObject(camera);

    auto floor = std::make_shared<Floor>(100.0f);
    render_system->AddRenderObject(floor->GetMesh());

    auto venus = MeshLoader::LoadMesh(std::string(VIEWER_DIR) + "/data/objs/venus.obj", 0.01f, glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(glm::radians(-90.0f), 0.0f, 0.0f), std::vector<glm::vec3>({glm::vec3(1.0f), glm::vec3(0.1f)}));
    venus->AddRenderer(std::make_shared<SolidColorRenderer>(glm::vec3(1.0f, 0.0f, 0.0f), true)); // Edge renderer
    render_system->AddRenderObject(venus);

    auto squirel = MeshLoader::LoadMesh(std::string(VIEWER_DIR) + std::string("/data/objs/CartoonSquirelModel/CartoonSquirelModel.obj"), 0.5f, glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f),
                                        std::string(VIEWER_DIR) + std::string("/data/objs/CartoonSquirelModel/DiffSqurel.tga"));
    render_system->AddRenderObject(squirel);

    auto dragon = MeshLoader::LoadMesh(std::string(VIEWER_DIR) + "/data/objs/dragon.obj", 6.0f, glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f), {glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.1, 0.1, 0.1)});
    render_system->AddRenderObject(dragon);

    auto light_scene = std::make_shared<LightScene>();
    app->AddObject(light_scene);

    auto lines = std::make_shared<LineSegment>(std::vector<glm::vec3>({glm::vec3(-1.0f, 0.0f, -1.0f), glm::vec3(-1.0f, 0.0f, 1.0f), glm::vec3(1.0f, 0.0f, -1.0f), glm::vec3(1.0f, 0.0f, 1.0f), glm::vec3(-1.0f, 2.0f, -1.0f), glm::vec3(-1.0f, 2.0f, 1.0f), glm::vec3(1.0f, 2.0f, -1.0f), glm::vec3(1.0f, 2.0f, 1.0f)}), std::vector<std::pair<int, int>>({{0, 4}, {1, 5}, {2, 6}, {3, 7}, {0, 1}, {0, 2}, {1, 3}, {2, 3}, {4, 5}, {4, 6}, {5, 7}, {6, 7}}));
    lines->AddRenderer(std::make_shared<SolidColorRenderer>(glm::vec3(1.0f, 1.0f, 1.0f), true));
    render_system->AddRenderObject(lines);

    auto particles = std::make_shared<ParticleBatch>(std::vector<Particle>({{glm::vec3(0.0f, 4.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f)}, {glm::vec3(0.0f, 5.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)}, {glm::vec3(0.0f, 6.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)}}));
    particles->AddRenderer(std::make_shared<SphereRenderer>(glm::vec2(0.1f), 0.2f));
    render_system->AddRenderObject(particles);

    app->Run();

    return 0;
}

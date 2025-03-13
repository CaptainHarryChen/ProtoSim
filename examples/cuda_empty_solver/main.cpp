#include <string>
#include <memory>
#include <GLFWApp.h>
#include <Render/RenderSystem.h>
#include <Scene/SimulationScene.h>
#include <Mesh/MeshLoader.h>
#include <Mesh/Mesh.h>
#include <Solver/EmptySolver.cuh>

int main()
{
    using Real = float;

    auto app = GLFWApp::GetInstance("Empty Solver Example", 1600, 900);
    auto render_system = app->GetRenderSystem();

    auto scene = std::make_shared<SimulationScene<Real>>();
    scene->SetupScene();
    app->AddObject(scene);

    scene->AddMesh(MeshLoader::LoadMesh(std::string(VIEWER_DIR) + "/data/objs/cube.obj", 1.0f, glm::vec3(0.0f, 10.0f, 0.0f), glm::vec3(0.0f), {glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.1, 0.1, 0.1)}));
    scene->AddMesh(MeshLoader::LoadMesh(std::string(VIEWER_DIR) + "/data/objs/dragon.obj", 6.0f, glm::vec3(1.0f, 5.0f, 0.0f), glm::vec3(0.0f), {glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.1, 0.1, 0.1)}));
    scene->AddMesh(MeshLoader::LoadMesh(std::string(VIEWER_DIR) + std::string("/data/objs/CartoonSquirelModel/CartoonSquirelModel.obj"), 0.5f, glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f),
                                        std::string(VIEWER_DIR) + std::string("/data/objs/CartoonSquirelModel/DiffSqurel.tga")));

    auto solver = std::make_shared<EmptySolver<Real>>(scene->m_positions);
    scene->SetSolver(solver);
    scene->SetupConnectors();

    app->Run();

    return 0;
}

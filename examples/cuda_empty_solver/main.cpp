#include <string>
#include <memory>
#include <GLFWApp.h>
#include <Render/RenderSystem.h>
#include <Scene/SimulationScene.h>
#include <Mesh/MeshLoader.h>
#include <Mesh/Mesh.h>
#include <Connector/MeshConnector.cuh>
#include <Solver/EmptySolver.cuh>

int main()
{
    using Real = float;

    auto app = GLFWApp::GetInstance("Empty Solver Example", 1600, 900);
    auto render_system = app->GetRenderSystem();

    auto scene = std::make_shared<SimulationScene<Real>>();
    scene->SetupScene();
    app->AddObject(scene);

    auto cube = MeshLoader::LoadMesh(std::string(VIEWER_DIR) + "/data/objs/cube.obj", 1.0f, glm::vec3(0.0f, 10.0f, 0.0f), glm::vec3(0.0f), {glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.1, 0.1, 0.1)});
    render_system->AddRenderObject(cube);
    auto dragon = MeshLoader::LoadMesh(std::string(VIEWER_DIR) + "/data/objs/dragon.obj", 6.0f, glm::vec3(1.0f, 5.0f, 0.0f), glm::vec3(0.0f), {glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.1, 0.1, 0.1)});
    render_system->AddRenderObject(dragon);

    std::vector<std::shared_ptr<Mesh>> meshes = {cube, dragon};

    std::vector<Real> positions;
    std::vector<size_t> offsets;
    size_t current_offset = 0;
    for(auto& mesh : meshes)
    {
        offsets.push_back(current_offset);
        for(auto &vert : mesh->m_vertices)
        {
            positions.push_back(vert.position.x);
            positions.push_back(vert.position.y);
            positions.push_back(vert.position.z);
        }
        current_offset += mesh->m_vertices.size() * 3;
    }

    auto solver = std::make_shared<EmptySolver<Real>>(positions);
    scene->SetSolver(solver);

    auto &connectors = scene->GetConnectors();
    for (size_t i = 0; i < meshes.size(); i++)
    {
        connectors.push_back(std::make_shared<MeshConnector<Real>>(meshes[i], solver->m_data.dev_position + offsets[i]));
    }

    app->Run();

    return 0;
}

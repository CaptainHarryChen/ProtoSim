#include <string>
#include <vector>
#include <memory>
#include <GLFWApp.h>
#include <Scene/SimulationScene.h>
#include <Loader/TetrahedronLoader.h>
#include <Mesh/Mesh.h>
#include <Mesh/PbrRenderer.h>
#include <Solver/ProjectiveDynamicsSolver.cuh>
#include <proj_config.h>

template <typename Real>
class TetrahedronScene : public SimulationScene<Real>
{
public:
    // std::vector<Real> m_positions;
    std::vector<unsigned int> m_surface_triangles;
    std::vector<unsigned int> m_tetrahedras;

    void LoadTetrahedron(std::string inputfile,
                         float scale, glm::vec3 translate, glm::vec3 rotate,
                         std::vector<glm::vec3> material)
    {
        std::vector<float> positions;
        std::vector<unsigned int> surface_triangles;
        std::vector<unsigned int> tetrahedras;
        TetrahedronLoader::LoadTetrahedron(inputfile, scale, translate, rotate, positions, surface_triangles, tetrahedras);

        std::vector<Vertex> vertices(positions.size() / 3);
        for (size_t i = 0; i < positions.size() / 3; ++i)
        {
            vertices[i].position = glm::vec3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
            vertices[i].normal = glm::vec3(0.0f, 0.0f, 0.0f);
            vertices[i].tex_coords = glm::vec2(0.0f, 0.0f);
        }
        auto mesh = std::make_shared<Mesh>(vertices, surface_triangles);
        mesh->AddRenderer(std::make_shared<PbrRenderer>(material));
        SimulationScene<Real>::AddMesh(mesh);

        unsigned int offset = (unsigned int)SimulationScene<Real>::m_mesh_offsets.back().second / 3;
        m_surface_triangles.resize(m_surface_triangles.size() + surface_triangles.size());
        for (size_t i = 0; i < surface_triangles.size(); ++i)
            m_surface_triangles[i + offset * 3] = surface_triangles[i] + offset;
        m_tetrahedras.resize(m_tetrahedras.size() + tetrahedras.size());
        for (size_t i = 0; i < tetrahedras.size(); ++i)
            m_tetrahedras[i + offset * 4] = tetrahedras[i] + offset;
    }
};

int main()
{
    using Real = float;

    auto app = GLFWApp::GetInstance("Projective Dynamics Solver Example", 1600, 900);

    auto scene = std::make_shared<TetrahedronScene<Real>>();
    scene->SetupScene();
    app->AddObject(scene);

    scene->LoadTetrahedron(std::string(ASSET_DIR) + "/bunny.tet",
                           10.0f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                           {glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.1f, 0.1f, 0.1f)});

    auto solver = std::make_shared<ProjectiveDynamicsSolver<Real>>();
    scene->SetSolver(solver);
    scene->SetupConnectors();

    app->Run();

    return 0;
}

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
    std::vector<unsigned int> m_tetrahedras;
    std::vector<Real> m_tetrahedras_densities;

    void LoadTetrahedron(std::string inputfile, Real density,
                         float scale, glm::vec3 translate, glm::vec3 rotate,
                         std::vector<glm::vec3> render_material)
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
        mesh->AddRenderer(std::make_shared<PbrRenderer>(render_material));
        SimulationScene<Real>::AddMesh(mesh);

        unsigned int offset = (unsigned int)SimulationScene<Real>::m_mesh_offsets.back().second / 3;
        m_tetrahedras.resize(m_tetrahedras.size() + tetrahedras.size());
        for (size_t i = 0; i < tetrahedras.size(); ++i)
            m_tetrahedras[i + offset * 4] = tetrahedras[i] + offset;
        m_tetrahedras_densities.resize(m_tetrahedras.size() / 4);
        for (size_t i = 0; i < m_tetrahedras_densities.size(); ++i)
            m_tetrahedras_densities[i] = density;
    }
};

int main()
{
    using Real = float;

    auto app = GLFWApp::GetInstance("Projective Dynamics Solver Example", 1600, 900);

    auto scene = std::make_shared<TetrahedronScene<Real>>();
    scene->SetupScene();
    app->AddObject(scene);

    scene->LoadTetrahedron(std::string(ASSET_DIR) + "/bunny", 1000.0f,
                           10.0f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(glm::radians(-45.0f), 0.0f, 0.0f),
                           {glm::vec3(0.5f, 0.5f, 1.0f), glm::vec3(0.1f, 0.1f, 0.1f)});
    // scene->LoadTetrahedron(std::string(ASSET_DIR) + "/armadillo10K", 1000.0f,
    //                        0.02f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(glm::radians(-45.0f), 0.0f, 0.0f),
    //                        {glm::vec3(0.5f, 0.5f, 1.0f), glm::vec3(0.1f, 0.1f, 0.1f)});

    auto solver = std::make_shared<ProjectiveDynamicsSolver<Real>>(scene->m_positions, scene->m_tetrahedras, scene->m_tetrahedras_densities);
    scene->SetSolver(solver);
    scene->SetupConnectors();

    app->Run();

    return 0;
}

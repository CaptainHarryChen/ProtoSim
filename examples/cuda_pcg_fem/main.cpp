#include <any>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <common.h>
#include <Solver/PCGFEMSolver.cuh>

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
        viewer::TetrahedronLoader::LoadTetrahedron(inputfile, scale, translate, rotate, positions, surface_triangles, tetrahedras);

        std::vector<viewer::Vertex> vertices(positions.size() / 3);
        for (size_t i = 0; i < positions.size() / 3; ++i)
        {
            vertices[i].position = glm::vec3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
            vertices[i].normal = glm::vec3(0.0f, 0.0f, 0.0f);
            vertices[i].tex_coords = glm::vec2(0.0f, 0.0f);
        }
        auto mesh = std::make_shared<viewer::Mesh>(vertices, surface_triangles);
        mesh->AddRenderer(std::make_shared<viewer::PbrRenderer>(render_material));
        SimulationScene<Real>::AddMesh(mesh);

        unsigned int node_offset = (unsigned int)SimulationScene<Real>::m_mesh_offsets.back().second / 3;
        unsigned int tet_offset = (unsigned int)m_tetrahedras.size() / 4;
        m_tetrahedras.resize(m_tetrahedras.size() + tetrahedras.size());
        for (size_t i = 0; i < tetrahedras.size(); ++i)
            m_tetrahedras[i + tet_offset * 4] = tetrahedras[i] + node_offset;
        m_tetrahedras_densities.resize(m_tetrahedras_densities.size() + m_tetrahedras.size() / 4);
        for (size_t i = 0; i < m_tetrahedras.size() / 4; ++i)
            m_tetrahedras_densities[i + tet_offset] = density;
    }

    virtual void SetupScene() override
    {
        SimulationScene<Real>::SetupScene();
        auto app = viewer::GLFWApp::GetInstance();
        for (auto &object : app->m_objects)
        {
            auto lightScene = std::dynamic_pointer_cast<viewer::LightScene>(object);
            if (lightScene)
            {
                lightScene->m_control_gui->m_light_on = {true, true, true, true};
                lightScene->m_control_gui->m_light_pos = {
                    glm::vec3(-10.0f, 10.0f, -10.0f),
                    glm::vec3(-10.0f, 10.0f, 10.0f),
                    glm::vec3(10.0f, 10.0f, -10.0f),
                    glm::vec3(10.0f, 10.0f, 10.0f)};
                break;
            }
        }
    }
};

int main()
{
    // Corotated Linear Model can use float precision
    // Neohookean Model requires double precision for better stability
    using Real = float;

    auto app = viewer::GLFWApp::GetInstance("PCG FEM Solver Example", 1600, 900);

    auto scene = std::make_shared<TetrahedronScene<Real>>();
    scene->SetupScene();
    app->AddObject(scene);

    // scene->LoadTetrahedron(std::string(ASSET_DIR) + "/bunny", 1000.0f,
    //                        15.0f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(glm::radians(-45.0f), 0.0f, 0.0f),
    //                        {glm::vec3(0.5f, 0.5f, 1.0f), glm::vec3(0.1f, 0.1f, 0.1f)});
    scene->LoadTetrahedron(std::string(ASSET_DIR) + "/armadillo/armadillo10K", 1000.0f,
                           0.12f, glm::vec3(0.0f, 5.5f, 0.0f), glm::vec3(glm::radians(-45.0f), 0.0f, 0.0f),
                           {glm::vec3(0.5f, 0.5f, 1.0f), glm::vec3(0.1f, 0.1f, 0.1f)});
    // scene->LoadTetrahedron(std::string(ASSET_DIR) + "/sphere/sphere1.5k", 1000.0f,
    //                        1.0f, glm::vec3(0.0f, 2.0f, 0.0f), glm::vec3(glm::radians(-45.0f), 0.0f, 0.0f),
    //                        {glm::vec3(0.5f, 0.5f, 1.0f), glm::vec3(0.1f, 0.1f, 0.1f)});
    // scene->LoadTetrahedron(std::string(ASSET_DIR) + "/cube/cube1.5k", 1000.0f,
    //                        1.0f, glm::vec3(0.0f, 2.0f, 0.0f), glm::vec3(glm::radians(-45.0f), 0.0f, 0.0f),
    //                        {glm::vec3(0.5f, 0.5f, 1.0f), glm::vec3(0.1f, 0.1f, 0.1f)});
    // scene->LoadTetrahedron(std::string(ASSET_DIR) + "/cube/cube6k", 1000.0f,
    //                        1.0f, glm::vec3(0.0f, 2.0f, 0.0f), glm::vec3(glm::radians(-45.0f), 0.0f, 0.0f),
    //                        {glm::vec3(0.5f, 0.5f, 1.0f), glm::vec3(0.1f, 0.1f, 0.1f)});

    Real young_k = 1000000.0f, young_nu = 0.26f;
    std::unordered_map<std::string, std::any> config {
        {"lame_mu", (Real)(young_k / (2 * (1 + young_nu)))},
        {"lame_lambda", (Real)(young_k * young_nu / ((1 + young_nu) * (1 - 2 * young_nu)))},
        {"ground_collision_stiffness", (Real)100000.0f},
        {"gravity", std::vector<Real>{0.0f, -9.81f, 0.0f}},
        {"time_step", (Real)(1.0f / 60.0f)},
        {"line_search_max_iteration", (unsigned int)3},
        {"fem_pcg_max_iteration", (unsigned int)300},
        {"fem_pcg_residual_tolerance", (Real)1e-1f}
    };

    auto solver = std::make_shared<PCGFEMSolver<Real>>(
        scene->m_positions,
        scene->m_tetrahedras,
        scene->m_tetrahedras_densities,
        config
    );
    solver->m_verbose = true;
    scene->SetSolver(solver);
    scene->SetupConnectors();

    app->Run();

    return 0;
}

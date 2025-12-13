#include <string>
#include <vector>
#include <memory>
#include <Eigen/Core>
#include <igl/random_points_on_mesh.h>
#include <GLFWApp.h>
#include <Scene/SimulationScene.h>
#include <Loader/TetrahedronLoader.h>
#include <Object/CubeLineBox.h>
#include <Object/LightScene.h>
#include <Mesh/Mesh.h>
#include <Mesh/PbrRenderer.h>
#include <Geometry/ParticleBatch.h>
#include <Geometry/LineSegment.h>
#include <Geometry/SphereRenderer.h>
#include <Mesh/SolidColorRenderer.h>
#include <Render/RenderSystem.h>
#include <Connector/MeshConnector.cuh>
#include <Connector/ParticleConnector.cuh>
#include <Solver/PCGCoupledMPMSolver.cuh>
#include <proj_config.h>
#include "GridTriangleDebugConnector.cuh"

template <typename Real>
class PCGCoupledMPMScene : public SimulationScene<Real>
{
public:
    // std::vector<Real> m_positions;
    std::vector<unsigned int> m_tetrahedras;
    std::vector<Real> m_tetrahedras_densities;
    std::vector<unsigned int> m_surface_triangles;

    std::vector<Real> m_sample_barycentric_weights;
    std::vector<unsigned int> m_sample_tri_idx;
    std::shared_ptr<ParticleBatch> m_sample_particle_batch;
    std::vector<std::pair<std::shared_ptr<ParticleBatch>, size_t>> m_sample_batch_offsets;

    std::vector<Real> m_particle_positions;
    std::vector<unsigned int> m_particle_types;
    std::vector<Real> m_particle_masses;
    std::vector<Real> m_particle_volumes;

    void LoadTetrahedronWithPLYSample(std::string inputfile, Real density, Real per_sample_volume,
                                      float scale, glm::vec3 translate, glm::vec3 rotate,
                                      std::vector<glm::vec3> mesh_render_material,
                                      float sample_radius, glm::vec3 sample_color, glm::vec2 sample_material)
    {
        std::vector<float> positions;
        std::vector<unsigned int> surface_triangles;
        std::vector<unsigned int> tetrahedras;
        std::vector<unsigned int> sample_tri_idx;
        std::vector<float> sample_barycentric_weights;
        TetrahedronLoader::LoadTetrahedronWithPLYSample(inputfile, scale, translate, rotate, positions, surface_triangles, tetrahedras, sample_tri_idx, sample_barycentric_weights);

        std::vector<Vertex> vertices(positions.size() / 3);
        for (size_t i = 0; i < positions.size() / 3; ++i)
        {
            vertices[i].position = glm::vec3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
            vertices[i].normal = glm::vec3(0.0f, 0.0f, 0.0f);
            vertices[i].tex_coords = glm::vec2(0.0f, 0.0f);
        }
        auto mesh = std::make_shared<Mesh>(vertices, surface_triangles);
        mesh->AddRenderer(std::make_shared<PbrRenderer>(mesh_render_material));
        mesh->AddRenderer(std::make_shared<SolidColorRenderer>(glm::vec3(0.0f, 0.0f, 0.0f), true));
        SimulationScene<Real>::AddMesh(mesh);

        unsigned int node_offset = (unsigned int)SimulationScene<Real>::m_mesh_offsets.back().second / 3;
        unsigned int tet_offset = (unsigned int)m_tetrahedras.size() / 4;
        m_tetrahedras.resize(m_tetrahedras.size() + tetrahedras.size());
        for (size_t i = 0; i < tetrahedras.size(); ++i)
            m_tetrahedras[i + tet_offset * 4] = tetrahedras[i] + node_offset;
        m_tetrahedras_densities.resize(m_tetrahedras_densities.size() + m_tetrahedras.size() / 4);
        for (size_t i = 0; i < m_tetrahedras.size() / 4; ++i)
            m_tetrahedras_densities[i + tet_offset] = density;

        unsigned int tri_offset = (unsigned int)m_surface_triangles.size() / 3;
        m_surface_triangles.resize(m_surface_triangles.size() + surface_triangles.size());
        for (size_t i = 0; i < surface_triangles.size(); ++i)
            m_surface_triangles[i + tri_offset * 3] = surface_triangles[i] + node_offset;

        unsigned int sample_offset = (unsigned int)m_sample_tri_idx.size();
        m_sample_tri_idx.resize(m_sample_tri_idx.size() + sample_tri_idx.size());
        for (size_t i = 0; i < sample_tri_idx.size(); ++i)
            m_sample_tri_idx[i + sample_offset] = sample_tri_idx[i] + tri_offset;

        m_sample_barycentric_weights.resize(sample_offset * 3 + sample_barycentric_weights.size());
        for (size_t i = 0; i < sample_barycentric_weights.size(); ++i)
            m_sample_barycentric_weights[i + sample_offset * 3] = sample_barycentric_weights[i];

        std::vector<Particle> particles(sample_tri_idx.size());
        for (size_t i = 0; i < sample_tri_idx.size(); ++i)
        {
            unsigned int tri_idx = sample_tri_idx[i];
            glm::vec3 bary_coords = glm::vec3(
                sample_barycentric_weights[i * 3 + 0],
                sample_barycentric_weights[i * 3 + 1],
                sample_barycentric_weights[i * 3 + 2]);
            glm::vec3 v0 = vertices[surface_triangles[tri_idx * 3 + 0]].position;
            glm::vec3 v1 = vertices[surface_triangles[tri_idx * 3 + 1]].position;
            glm::vec3 v2 = vertices[surface_triangles[tri_idx * 3 + 2]].position;
            particles[i].Position = bary_coords.x * v0 + bary_coords.y * v1 + bary_coords.z * v2;
            particles[i].Color = sample_color;
        }
        m_sample_particle_batch = std::make_shared<ParticleBatch>(particles);
        m_sample_batch_offsets.push_back(std::make_pair(m_sample_particle_batch, sample_offset * 3));
        m_sample_particle_batch->AddRenderer(std::make_shared<SphereRenderer>(sample_material, sample_radius));
        GLFWApp::GetInstance()->GetRenderSystem()->AddRenderObject(m_sample_particle_batch);
    }

    void AddMPMCubeParticleBatch(glm::vec3 lower_bound, glm::vec3 upper_bound, float dis,
                                 unsigned int particle_type, Real density,
                                 float radius, glm::vec3 color, glm::vec2 material)
    {
        std::vector<Particle> particles;
        size_t node_offset = m_particle_positions.size();
        for (Real x = lower_bound.x; x <= upper_bound.x; x += dis)
            for (Real y = lower_bound.y; y <= upper_bound.y; y += dis)
                for (Real z = lower_bound.z; z <= upper_bound.z; z += dis)
                {
                    particles.push_back({glm::vec3(x, y, z), color});
                    m_particle_positions.push_back(x);
                    m_particle_positions.push_back(y);
                    m_particle_positions.push_back(z);
                }
        auto particle_batch = std::make_shared<ParticleBatch>(particles);
        this->m_particle_batch_offsets.push_back(std::make_pair(particle_batch, node_offset));
        particle_batch->AddRenderer(std::make_shared<SphereRenderer>(material, radius));
        GLFWApp::GetInstance()->GetRenderSystem()->AddRenderObject(particle_batch);

        size_t num_particle = this->m_particle_batch_offsets.back().first->m_particles.size();
        Real total_volume = (upper_bound.x - lower_bound.x) * (upper_bound.y - lower_bound.y) * (upper_bound.z - lower_bound.z);
        Real particle_volume = dis * dis * dis;
        Real particle_mass = density * particle_volume;
        for (size_t i = 0; i < num_particle; ++i)
        {
            m_particle_types.push_back(particle_type);
            m_particle_masses.push_back(particle_mass);
            m_particle_volumes.push_back(particle_volume);
        }
    }

    virtual void SetupConnectors() override
    {
        auto solver = std::dynamic_pointer_cast<PCGCoupledMPMSolver<Real>>(this->m_solver);
        assert(solver != nullptr && "Solver must be PCGCoupledMPMSolver");
        for (auto &[mesh, node_offset] : this->m_mesh_offsets)
        {
            auto position_ptr = solver->GetDeviceVertexPositions();
            if (position_ptr)
                position_ptr = position_ptr + node_offset;
            auto connector = std::make_shared<MeshConnector<Real>>(mesh, position_ptr);
            this->m_connectors.push_back(connector);
        }
        for (auto &[sample, node_offset] : this->m_sample_batch_offsets)
        {
            auto position_ptr = solver->GetDeviceSamplePositions();
            if (position_ptr)
                position_ptr = position_ptr + node_offset;
            auto connector = std::make_shared<ParticleConnector<Real>>(sample, position_ptr, nullptr);
            this->m_connectors.push_back(connector);
        }
        for (auto &[particle_batch, node_offset] : this->m_particle_batch_offsets)
        {
            auto position_ptr = solver->GetDeviceParticlePositions();
            auto color_ptr = solver->m_data.dev_particle_color;
            if (position_ptr)
                position_ptr = position_ptr + node_offset;
            if (color_ptr)
                color_ptr = color_ptr + node_offset;
            auto connector = std::make_shared<ParticleConnector<Real>>(particle_batch, position_ptr, color_ptr);
            this->m_connectors.push_back(connector);
        }
    }

    virtual void SetupScene() override
    {
        SimulationScene<Real>::SetupScene();
        auto app = GLFWApp::GetInstance();
        for (auto &object : app->m_objects)
        {
            auto lightScene = std::dynamic_pointer_cast<LightScene>(object);
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
    using Real = float;

    auto app = GLFWApp::GetInstance("PCG Coupled MPM Solver Example", 1600, 900);

    auto scene = std::make_shared<PCGCoupledMPMScene<Real>>();
    scene->SetupScene();
    app->AddObject(scene);

    // scene->LoadTetrahedronWithPLYSample(std::string(ASSET_DIR) + "/bunny", 1000.0f, 0.039f * 0.039f * 0.039f,
    //                                     15.0f, glm::vec3(0.0f, 4.0f, 2.0f), glm::vec3(glm::radians(-45.0f), 0.0f, 0.0f),
    //                                     {glm::vec3(1.0f, 0.5f, 1.0f), glm::vec3(0.1f, 0.1f, 0.1f)},
    //                                     0.01f, glm::vec3(1.0f, 0.0f, 0.0f), glm::vec2(0.1f, 0.1f));

    scene->AddMPMCubeParticleBatch(glm::vec3(-2.0f, 6.1f, -2.0f), glm::vec3(2.0f, 7.1f, 2.0f), 0.08f,
                                   MPM_FLUID, 100.0f,
                                   0.03f, glm::vec3(0.2f, 0.2f, 1.0f), glm::vec2(0.8f, 0.8f));
    scene->LoadTetrahedronWithPLYSample(std::string(ASSET_DIR) + "/bunny", 1000.0f, 0.039f * 0.039f * 0.039f,
                                        15.0f, glm::vec3(0.0f, 0.1f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                                        {glm::vec3(1.0f, 1.0f, 0.5f), glm::vec3(0.1f, 0.1f, 0.1f)},
                                        0.005f, glm::vec3(1.0f, 0.0f, 0.0f), glm::vec2(0.1f, 0.1f));

    float dist = 0.2f;
    // std::vector<float> bbox = {-10.0f, 0.0f, -10.0f, 10.0f, 20.0f, 10.0f};
    std::vector<float> bbox = {-5.0f, 0.0f, -5.0f, 5.0f, 10.0f, 5.0f};
    unsigned int boundary_thickness = 1;
    app->AddObject(std::make_shared<CubeLineBox>(bbox, dist, glm::vec3(1.0f, 1.0f, 1.0f)));
    std::vector<Real> real_bbox;
    for (auto b : bbox)
        real_bbox.push_back((Real)b);

    Real young_k = 1000000.0f, young_nu = 0.26f;
    std::unordered_map<std::string, std::any> config {
        {"lame_mu", (Real)(young_k / (2 * (1 + young_nu)))},
        {"lame_lambda", (Real)(young_k * young_nu / ((1 + young_nu) * (1 - 2 * young_nu)))},
        {"fluid_lambda", (Real)(1000000.0f)},
        {"fluid_viscosity", (Real)(1.0f)},
        {"ground_collision_stiffness", (Real)100000.0f},
        {"contact_stiffness", (Real)100000000.0f},
        {"gravity", std::vector<Real>{0.0f, -9.81f, 0.0f}},
        {"time_step", (Real)(1.0f / 1000.0f)},
        {"fem_pcg_max_iteration", (unsigned int)30},
        {"fem_pcg_residual_tolerance", (Real)1e-2f},
        {"mpm_pcg_max_iteration", (unsigned int)300},
        {"mpm_pcg_residual_tolerance", (Real)1e-2f}
    };

    auto solver = std::make_shared<PCGCoupledMPMSolver<Real>>(
        scene->m_positions,
        scene->m_surface_triangles,
        scene->m_tetrahedras,
        scene->m_tetrahedras_densities,

        scene->m_sample_barycentric_weights,
        scene->m_sample_tri_idx,

        scene->m_particle_positions,
        scene->m_particle_types,
        scene->m_particle_masses,
        scene->m_particle_volumes,

        real_bbox, dist, boundary_thickness,
        config
    );
    solver->m_verbose = true;
    scene->SetSolver(solver);
    scene->SetupConnectors();
    scene->SetStepPerFrame(1);

    // unsigned int num_grid = solver->m_mpm_solver.m_data.m_num_grid;
    // auto grid_inside_monitor = std::make_shared<ParticleBatch>(std::vector<Particle>(num_grid));
    // grid_inside_monitor->AddRenderer(std::make_shared<SphereRenderer>(glm::vec3(0.1f, 0.1f, 0.1f), 0.02f));
    // app->GetRenderSystem()->AddRenderObject(grid_inside_monitor);
    // std::vector<LineSeg> line_segs(num_grid);
    // auto grid_dis_monitor = std::make_shared<LineSegment>(line_segs);
    // grid_dis_monitor->AddRenderer(std::make_shared<SolidColorRenderer>(glm::vec3(1.0f, 0.0f, 0.0f)));
    // app->GetRenderSystem()->AddRenderObject(grid_dis_monitor);
    // scene->AddConnector(std::make_shared<GridTriangleDebugConnector<Real>>(grid_inside_monitor, grid_dis_monitor, &solver->m_data, solver->m_dev_data));

    app->Run();

    return 0;
}

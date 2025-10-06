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
#include <Geometry/SphereRenderer.h>
#include <Render/RenderSystem.h>
#include <Connector/MeshConnector.cuh>
#include <Connector/ParticleConnector.cuh>
#include <Solver/ImplicitCPICSolver.cuh>
#include <proj_config.h>
// #include "GridTriangleDebugConnector.cuh"

template <typename Real>
class CPICScene : public SimulationScene<Real>
{
public:
    // std::vector<Real> m_positions;
    std::vector<unsigned int> m_tetrahedras;
    std::vector<Real> m_tetrahedras_densities;
    std::vector<unsigned int> m_surface_triangles;

    std::vector<Real> m_sample_barycentric_weights;
    std::vector<unsigned int> m_sample_tri_idx;
    std::shared_ptr<ParticleBatch> m_sample_particle_batch;

    std::vector<Real> m_particle_positions;
    std::vector<unsigned int> m_particle_types;
    std::vector<Real> m_particle_masses;
    std::vector<Real> m_particle_volumes;

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
    }

    void AddMPMCubeParticleBatch(glm::vec3 lower_bound, glm::vec3 upper_bound, float dis,
                                 unsigned int particle_type, Real density,
                                 float radius, glm::vec3 color, glm::vec2 material)
    {
        std::vector<Particle> particles;
        size_t offset = m_particle_positions.size();
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
        this->m_particle_batch_offsets.push_back(std::make_pair(particle_batch, offset));
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

    void SampleSurfaceParticles(int num_samples, float radius, glm::vec3 color, glm::vec2 material)
    {
        Eigen::MatrixXd V(this->m_positions.size() / 3, 3);
        for (size_t i = 0; i < this->m_positions.size() / 3; ++i)
        {
            V(i, 0) = this->m_positions[i * 3 + 0];
            V(i, 1) = this->m_positions[i * 3 + 1];
            V(i, 2) = this->m_positions[i * 3 + 2];
        }
        Eigen::MatrixXi F(m_surface_triangles.size() / 3, 3);
        for (size_t i = 0; i < m_surface_triangles.size() / 3; ++i)
        {
            F(i, 0) = m_surface_triangles[i * 3 + 0];
            F(i, 1) = m_surface_triangles[i * 3 + 1];
            F(i, 2) = m_surface_triangles[i * 3 + 2];
        }

        Eigen::MatrixXd samples;
        Eigen::MatrixXd bary_coords;
        Eigen::VectorXi face_indices;
        igl::random_points_on_mesh(
            num_samples,
            V, F,
            bary_coords,
            face_indices,
            samples //
        );

        m_sample_barycentric_weights.resize(bary_coords.size());
        for (int i = 0; i < bary_coords.rows(); ++i)
        {
            m_sample_barycentric_weights[i * 3 + 0] = bary_coords(i, 0);
            m_sample_barycentric_weights[i * 3 + 1] = bary_coords(i, 1);
            m_sample_barycentric_weights[i * 3 + 2] = bary_coords(i, 2);
        }
        m_sample_tri_idx.resize(face_indices.size());
        for (int i = 0; i < face_indices.size(); ++i)
        {
            m_sample_tri_idx[i] = face_indices(i);
        }

        std::vector<Particle> particles(samples.rows());
        for (int i = 0; i < samples.rows(); ++i)
        {
            particles[i].Position = glm::vec3(samples(i, 0), samples(i, 1), samples(i, 2));
            particles[i].Color = color;
        }
        m_sample_particle_batch = std::make_shared<ParticleBatch>(particles);
        // m_sample_particle_batch->AddRenderer(std::make_shared<SphereRenderer>(material, radius));
        GLFWApp::GetInstance()->GetRenderSystem()->AddRenderObject(m_sample_particle_batch);
    }

    virtual void SetupConnectors() override
    {
        auto solver = std::dynamic_pointer_cast<ImplicitCPICSolver<Real>>(this->m_solver);
        assert(solver != nullptr && "Solver must be ImplicitCPICSolver");
        for (auto &[mesh, offset] : this->m_mesh_offsets)
        {
            auto position_ptr = solver->GetDeviceNodePositions();
            if (position_ptr)
                position_ptr = position_ptr + offset;
            auto connector = std::make_shared<MeshConnector<Real>>(mesh, position_ptr);
            this->m_connectors.push_back(connector);
        }
        {
            auto position_ptr = solver->GetDeviceSamplePositions();
            auto connector = std::make_shared<ParticleConnector<Real>>(m_sample_particle_batch, position_ptr, nullptr);
            this->m_connectors.push_back(connector);
        }
        for (auto &[particle_batch, offset] : this->m_particle_batch_offsets)
        {
            auto position_ptr = solver->GetDeviceParticlePositions();
            if (position_ptr)
                position_ptr = position_ptr + offset;
            auto connector = std::make_shared<ParticleConnector<Real>>(particle_batch, position_ptr, nullptr);
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
            }
        }
    }
};

int main()
{
    using Real = float;

    auto app = GLFWApp::GetInstance("Implicit CPIC Solver Example", 1600, 900);

    auto scene = std::make_shared<CPICScene<Real>>();
    scene->SetupScene();
    app->AddObject(scene);

    // scene->LoadTetrahedron(std::string(ASSET_DIR) + "/bunny", 1000.0f,
    //                        15.0f, glm::vec3(0.0f, 0.2f, 1.5f), glm::vec3(glm::radians(-45.0f), 0.0f, 0.0f),
    //                        {glm::vec3(1.0f, 1.0f, 0.5f), glm::vec3(0.1f, 0.1f, 0.1f)});
    // scene->AddMPMCubeParticleBatch(glm::vec3(-3.0f, 4.5f, -2.0f), glm::vec3(3.0f, 5.5f, 2.0f), 0.08f,
    //                                MPM_FLUID, 1000.0f,
    //                                0.03f, glm::vec3(0.2f, 0.2f, 1.0f), glm::vec2(0.8f, 0.8f));
    // scene->SampleSurfaceParticles(20000, 0.03f, glm::vec3(1.0f, 0.0f, 0.0f), glm::vec2(0.1f, 0.1f));

    scene->LoadTetrahedron(std::string(ASSET_DIR) + "/armadillo10K", 1000.0f,
                           0.1f, glm::vec3(0.0f, 5.0f, 2.0f), glm::vec3(glm::radians(-45.0f), 0.0f, 0.0f),
                           {glm::vec3(1.0f, 1.0f, 0.5f), glm::vec3(0.1f, 0.1f, 0.1f)});
    scene->AddMPMCubeParticleBatch(glm::vec3(-3.0f, 14.0f, -2.0f), glm::vec3(3.0f, 17.0f, 2.0f), 0.08f,
                                   MPM_FLUID, 1000.0f,
                                   0.03f, glm::vec3(0.2f, 0.2f, 1.0f), glm::vec2(0.8f, 0.8f));
    scene->SampleSurfaceParticles(40000, 0.03f, glm::vec3(1.0f, 0.0f, 0.0f), glm::vec2(0.1f, 0.1f));
    std::vector<float> bbox = {-10.0f, 0.0f, -10.0f, 10.0f, 20.0f, 10.0f};

    // scene->LoadTetrahedron(std::string(ASSET_DIR) + "/bunny", 100.0f,
    //                        20.0f, glm::vec3(0.0f, 3.5f, 2.0f), glm::vec3(glm::radians(-45.0f), 0.0f, 0.0f),
    //                        {glm::vec3(1.0f, 1.0f, 0.5f), glm::vec3(0.1f, 0.1f, 0.1f)});
    // scene->AddMPMCubeParticleBatch(glm::vec3(-4.9f, 0.1f, -4.9f), glm::vec3(4.9f, 3.1f, 4.9f), 0.08f,
    //                                MPM_FLUID, 1000.0f,
    //                                0.03f, glm::vec3(0.2f, 0.2f, 1.0f), glm::vec2(0.8f, 0.8f));
    // scene->SampleSurfaceParticles(10000, 0.03f, glm::vec3(1.0f, 0.0f, 0.0f), glm::vec2(0.1f, 0.1f));
    // std::vector<float> bbox = {-5.0f, 0.0f, -5.0f, 5.0f, 10.0f, 5.0f};

    float dist = 0.2f;
    // std::vector<float> bbox = {-10.0f, 0.0f, -10.0f, 10.0f, 20.0f, 10.0f};
    unsigned int boundary_thickness = 1;
    app->AddObject(std::make_shared<CubeLineBox>(bbox, dist, glm::vec3(1.0f, 1.0f, 1.0f)));

    auto solver = std::make_shared<ImplicitCPICSolver<Real>>(
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

        bbox, dist, boundary_thickness);
    scene->SetSolver(solver);
    scene->SetupConnectors();
    scene->SetStepPerFrame(1);

    // unsigned int num_grid = solver->m_data.m_num_grid;
    // auto grid_inside_monitor = std::make_shared<ParticleBatch>(std::vector<Particle>(num_grid));
    // grid_inside_monitor->AddRenderer(std::make_shared<SphereRenderer>(glm::vec3(0.1f, 0.1f, 0.1f), 0.07f));
    // app->GetRenderSystem()->AddRenderObject(grid_inside_monitor);
    // scene->AddConnector(std::make_shared<GridTriangleDebugConnector<Real>>(grid_inside_monitor, &solver->m_data, solver->m_dev_data));

    app->Run();

    return 0;
}

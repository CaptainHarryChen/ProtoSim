#include "SimulationScene.h"
#include <GLFWApp.h>
#include <Camera/OrbitCamera.h>
#include <Object/Floor.h>
#include <Object/LightScene.h>
#include <Geometry/ParticleBatch.h>
#include <Geometry/SphereRenderer.h>
#include <Render/RenderSystem.h>
#include <Solver/Solver.cuh>
#include <Connector/MeshConnector.cuh>
#include <Connector/ParticleConnector.cuh>

template <typename Real>
void SimulationScene<Real>::Update(double delta_time)
{
    m_solver->Step();

    for (auto &connector : m_connectors)
    {
        connector->TransferData();
    }
}

template <typename Real>
void SimulationScene<Real>::SetupScene()
{
    auto app = GLFWApp::GetInstance();
    std::shared_ptr<RenderSystem> render_system = app->GetRenderSystem();

    auto camera = std::make_shared<OrbitCamera>(glm::radians(-30.0f), glm::radians(-30.0f), 10.0f);
    app->SetCamera(camera);
    app->AddObject(camera);

    auto floor = std::make_shared<Floor>(100.0f);
    render_system->AddRenderObject(floor->GetMesh());

    auto light_scene = std::make_shared<LightScene>();
    app->AddObject(light_scene);
}

template <typename Real>
void SimulationScene<Real>::SetSolver(std::shared_ptr<Solver<Real>> solver)
{
    m_solver = solver;
}

template <typename Real>
void SimulationScene<Real>::AddConnector(std::shared_ptr<Connector> connector)
{
    m_connectors.push_back(connector);
}

template <typename Real>
void SimulationScene<Real>::SetupConnectors()
{
    for (auto &[mesh, offset] : m_mesh_offsets)
    {
        auto position_ptr = m_solver->GetDevicePositions();
        if (position_ptr)
            position_ptr = position_ptr + offset;
        auto connector = std::make_shared<MeshConnector<Real>>(mesh, position_ptr);
        m_connectors.push_back(connector);
    }
    for (auto &[particle_batch, offset] : m_particle_batche_offsets)
    {
        auto position_ptr = m_solver->GetDevicePositions();
        if (position_ptr)
            position_ptr = position_ptr + offset;
        auto color_ptr = m_solver->GetDeviceColors();
        if (color_ptr)
            color_ptr = color_ptr + offset;
        auto connector = std::make_shared<ParticleConnector<Real>>(particle_batch, position_ptr, color_ptr);
        m_connectors.push_back(connector);
    }
}

template <typename Real>
void SimulationScene<Real>::AddMesh(std::shared_ptr<Mesh> mesh)
{
    m_mesh_offsets.push_back(std::make_pair(mesh, m_positions.size()));
    for (auto &vert : mesh->m_vertices)
    {
        m_positions.push_back(vert.position.x);
        m_positions.push_back(vert.position.y);
        m_positions.push_back(vert.position.z);
    }
    GLFWApp::GetInstance()->GetRenderSystem()->AddRenderObject(mesh);
}

template <typename Real>
void SimulationScene<Real>::AddParticleBatch(std::shared_ptr<ParticleBatch> particle_batch)
{
    m_particle_batche_offsets.push_back(std::make_pair(particle_batch, m_positions.size()));
    for (auto &particle : particle_batch->m_particles)
    {
        m_positions.push_back(particle.Position.x);
        m_positions.push_back(particle.Position.y);
        m_positions.push_back(particle.Position.z);
    }
    GLFWApp::GetInstance()->GetRenderSystem()->AddRenderObject(particle_batch);
}

template <typename Real>
void SimulationScene<Real>::AddCubeParticleBatch(
    glm::vec3 lower_bound, glm::vec3 upper_bound, float dis,
    float radius, glm::vec3 color, glm::vec2 material)
{
    std::vector<Particle> particles;
    int num_particle = 0;
    size_t offset = m_positions.size();
    for (Real x = lower_bound.x; x <= upper_bound.x; x += dis)
        for (Real y = lower_bound.y; y <= upper_bound.y; y += dis)
            for (Real z = lower_bound.z; z <= upper_bound.z; z += dis)
            {
                particles.push_back({glm::vec3(x, y, z), color});
                m_positions.push_back(x);
                m_positions.push_back(y);
                m_positions.push_back(z);
                num_particle++;
            }
    auto particle_batch = std::make_shared<ParticleBatch>(particles);
    m_particle_batche_offsets.push_back(std::make_pair(particle_batch, offset));
    particle_batch->AddRenderer(std::make_shared<SphereRenderer>(material, radius));
    GLFWApp::GetInstance()->GetRenderSystem()->AddRenderObject(particle_batch);
}

template class SimulationScene<float>;
template class SimulationScene<double>;

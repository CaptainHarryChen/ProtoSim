#include "common.h"
#include <chrono>
#include <imgui.h>
#include <viewer/GUI/FPSMonitor.h>
#include <viewer/Object/OrbitCamera.h>
#include <viewer/Object/Floor.h>
#include <Solver/Solver.cuh>

template <typename Real>
void SimulationScene<Real>::Update(double delta_time)
{
    if (m_play)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (unsigned int i = 0; i < m_step_per_frame; ++i)
            m_solver->Step();
        m_solver->SynchronizeStep();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = duration_cast<std::chrono::microseconds>(end - start);
        m_fps_monitor->UpdateFPS((double)duration.count() / m_step_per_frame);

        for (auto &connector : m_connectors)
        {
            connector->TransferData();
        }
    }
}

template <typename Real>
void SimulationScene<Real>::ProcessEvent(const viewer::Event &event)
{
    if (event.type == viewer::EventType::Key)
    {
        if (event.key.action == GLFW_PRESS)
        {
            if (event.key.key == GLFW_KEY_SPACE)
            {
                m_play = !m_play;
            }
        }
    }
}

template <typename Real>
void SimulationScene<Real>::SetupScene()
{
    auto app = viewer::GLFWApp::GetInstance();
    std::shared_ptr<viewer::RenderSystem> render_system = app->GetRenderSystem();

    auto camera = std::make_shared<viewer::OrbitCamera>(glm::radians(-30.0f), glm::radians(-30.0f), 10.0f);
    app->SetCamera(camera);
    app->AddObject(camera);

    auto floor = std::make_shared<viewer::Floor>(100.0f);
    render_system->AddRenderObject(floor->GetMesh());

    auto light_scene = std::make_shared<viewer::LightScene>();
    app->AddObject(light_scene);

    m_fps_monitor = std::make_shared<viewer::FPSMonitor>();
    render_system->AddRenderObject(m_fps_monitor);
}

template <typename Real>
void SimulationScene<Real>::SetStepPerFrame(unsigned int step_per_frame)
{
    m_step_per_frame = step_per_frame;
}

template <typename Real>
void SimulationScene<Real>::SetSolver(std::shared_ptr<Solver<Real>> solver)
{
    m_solver = solver;
}

template <typename Real>
void SimulationScene<Real>::AddConnector(std::shared_ptr<viewer::Connector> connector)
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
        auto connector = std::make_shared<viewer::MeshConnector<Real>>(mesh, position_ptr);
        m_connectors.push_back(connector);
    }
    for (auto &[particle_batch, offset] : m_particle_batch_offsets)
    {
        auto position_ptr = m_solver->GetDevicePositions();
        if (position_ptr)
            position_ptr = position_ptr + offset;
        auto color_ptr = m_solver->GetDeviceColors();
        if (color_ptr)
            color_ptr = color_ptr + offset;
        auto connector = std::make_shared<viewer::ParticleConnector<Real>>(particle_batch, position_ptr, color_ptr);
        m_connectors.push_back(connector);
    }
}

template <typename Real>
void SimulationScene<Real>::AddMesh(std::shared_ptr<viewer::Mesh> mesh)
{
    m_mesh_offsets.push_back(std::make_pair(mesh, m_positions.size()));
    for (auto &vert : mesh->m_vertices)
    {
        m_positions.push_back(vert.position.x);
        m_positions.push_back(vert.position.y);
        m_positions.push_back(vert.position.z);
    }
    viewer::GLFWApp::GetInstance()->GetRenderSystem()->AddRenderObject(mesh);
}

template <typename Real>
void SimulationScene<Real>::AddParticleBatch(std::shared_ptr<viewer::ParticleBatch> particle_batch)
{
    m_particle_batch_offsets.push_back(std::make_pair(particle_batch, m_positions.size()));
    for (auto &particle : particle_batch->m_particles)
    {
        m_positions.push_back(particle.Position.x);
        m_positions.push_back(particle.Position.y);
        m_positions.push_back(particle.Position.z);
    }
    viewer::GLFWApp::GetInstance()->GetRenderSystem()->AddRenderObject(particle_batch);
}

template <typename Real>
void SimulationScene<Real>::AddCubeParticleBatch(
    glm::vec3 lower_bound, glm::vec3 upper_bound, float dis,
    float radius, glm::vec3 color, glm::vec2 material)
{
    std::vector<viewer::Particle> particles;
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
    auto particle_batch = std::make_shared<viewer::ParticleBatch>(particles);
    m_particle_batch_offsets.push_back(std::make_pair(particle_batch, offset));
    particle_batch->AddRenderer(std::make_shared<viewer::SphereRenderer>(material, radius));
    viewer::GLFWApp::GetInstance()->GetRenderSystem()->AddRenderObject(particle_batch);
}

template class SimulationScene<float>;
template class SimulationScene<double>;

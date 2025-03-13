#include "SimulationScene.h"
#include <GLFWApp.h>
#include <Camera/OrbitCamera.h>
#include <Object/Floor.h>
#include <Object/LightScene.h>
#include <Render/RenderSystem.h>
#include <Solver/Solver.cuh>
#include <Connector/MeshConnector.cuh>

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

    auto camera = std::make_shared<OrbitCamera>(glm::radians(-45.0f), glm::radians(-45.0f), 10.0f);
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
        auto connector = std::make_shared<MeshConnector<Real>>(mesh, m_solver->GetDevicePositions() + offset);
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
    // m_particle_batche_offsets.push_back(std::make_pair(particle_batch, m_positions.size()));
    // for (auto &particle : particle_batch->m_particles)
    // {
    //     m_positions.push_back(particle.Position.x);
    //     m_positions.push_back(particle.Position.y);
    //     m_positions.push_back(particle.Position.z);
    // }
    // GLFWApp::GetInstance()->GetRenderSystem()->AddRenderObject(particle_batch);
}

template class SimulationScene<float>;
template class SimulationScene<double>;

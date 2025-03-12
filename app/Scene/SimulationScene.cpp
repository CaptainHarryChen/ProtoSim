#include "SimulationScene.h"
#include <GLFWApp.h>
#include <Camera/OrbitCamera.h>
#include <Object/Floor.h>
#include <Object/LightScene.h>
#include <Render/RenderSystem.h>
#include <Solver/Solver.cuh>
#include <Connector/Connector.cuh>

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
std::vector<std::shared_ptr<Connector>> &SimulationScene<Real>::GetConnectors()
{
    return m_connectors;
}

template class SimulationScene<float>;
template class SimulationScene<double>;

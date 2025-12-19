#pragma once

#include <viewer/Connector/MeshConnector.cuh>
#include <viewer/Connector/ParticleConnector.cuh>
#include <viewer/Framework/RenderSystem.h>
#include <viewer/Framework/Object.h>
#include <viewer/GLFWApp.h>
#include <viewer/Object/CubeLineBox.h>
#include <viewer/Object/LightScene.h>
#include <viewer/Renderer/PbrRenderer.h>
#include <viewer/Renderer/SolidColorRenderer.h>
#include <viewer/Renderer/SphereRenderer.h>
#include <viewer/RenderObject/ParticleBatch.h>
#include <viewer/RenderObject/Mesh.h>
#include <viewer/RenderObject/LineSegment.h>
#include <viewer/utils/MeshLoader.h>
#include <viewer/utils/TetrahedronLoader.h>
#include "example_config.h"

#include <vector>
#include <memory>
#include <glm/glm.hpp>

template <typename Real>
class Solver;

namespace viewer
{
    struct Event;
    class FPSMonitor;
}

template <typename Real>
class SimulationScene : public viewer::Object
{
public:
    SimulationScene() = default;
    virtual ~SimulationScene() = default;

    virtual void Update(double delta_time) override;
    virtual void ProcessEvent(const viewer::Event &event) override;

    virtual void SetupScene();
    virtual void SetStepPerFrame(unsigned int step_per_frame);
    virtual void SetSolver(std::shared_ptr<Solver<Real>> solver);
    virtual void AddConnector(std::shared_ptr<viewer::Connector> connector);
    virtual void SetupConnectors();

    virtual void AddMesh(std::shared_ptr<viewer::Mesh> mesh);
    virtual void AddParticleBatch(std::shared_ptr<viewer::ParticleBatch> particle_batch);
    virtual void AddCubeParticleBatch(glm::vec3 lower_bound, glm::vec3 upper_bound, float dis,
                                      float radius, glm::vec3 color, glm::vec2 material);

    std::vector<Real> m_positions;

protected:
    std::vector<std::pair<std::shared_ptr<viewer::Mesh>, size_t>> m_mesh_offsets;
    std::vector<std::pair<std::shared_ptr<viewer::ParticleBatch>, size_t>> m_particle_batch_offsets;

    std::shared_ptr<Solver<Real>> m_solver;
    std::vector<std::shared_ptr<viewer::Connector>> m_connectors;

    std::shared_ptr<viewer::FPSMonitor> m_fps_monitor;

    bool m_play = false;
    unsigned int m_step_per_frame = 1;
};

extern template class SimulationScene<float>;
extern template class SimulationScene<double>;

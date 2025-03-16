#pragma once
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <Object/Object.h>

template <typename Real>
class Solver;
class Event;
class Connector;
class Mesh;
class ParticleBatch;

template <typename Real>
class SimulationScene : public Object
{
public:
    SimulationScene() = default;
    virtual ~SimulationScene() = default;

    virtual void Update(double delta_time) override;
    virtual void ProcessEvent(const Event &event) override;

    virtual void SetupScene();
    virtual void SetSolver(std::shared_ptr<Solver<Real>> solver);
    virtual void AddConnector(std::shared_ptr<Connector> connector);
    virtual void SetupConnectors();

    virtual void AddMesh(std::shared_ptr<Mesh> mesh);
    virtual void AddParticleBatch(std::shared_ptr<ParticleBatch> particle_batch);
    virtual void AddCubeParticleBatch(glm::vec3 lower_bound, glm::vec3 upper_bound, float dis,
                                      float radius, glm::vec3 color, glm::vec2 material);

    std::vector<Real> m_positions;

protected:
    std::vector<std::pair<std::shared_ptr<Mesh>, size_t>> m_mesh_offsets;
    std::vector<std::pair<std::shared_ptr<ParticleBatch>, size_t>> m_particle_batch_offsets;

    std::shared_ptr<Solver<Real>> m_solver;
    std::vector<std::shared_ptr<Connector>> m_connectors;

    bool m_play = false;
};

extern template class SimulationScene<float>;
extern template class SimulationScene<double>;

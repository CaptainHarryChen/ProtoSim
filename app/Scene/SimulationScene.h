#pragma once
#include <vector>
#include <memory>
#include <Object/Object.h>

template <typename Real>
class Solver;
class Connector;
class Mesh;
class ParticleBatch;

template <typename Real>
class SimulationScene : public Object
{
public:
    SimulationScene() = default;
    virtual ~SimulationScene() = default;

    void Update(double delta_time) override;

    void SetupScene();
    void SetSolver(std::shared_ptr<Solver<Real>> solver);
    void AddConnector(std::shared_ptr<Connector> connector);
    void SetupConnectors();

    void AddMesh(std::shared_ptr<Mesh> mesh);
    void AddParticleBatch(std::shared_ptr<ParticleBatch> particle_batch);

    std::vector<Real> m_positions;

protected:
    std::vector<std::pair<std::shared_ptr<Mesh>, size_t>> m_mesh_offsets;
    std::vector<std::pair<std::shared_ptr<ParticleBatch>, size_t>> m_particle_batche_offsets;

    std::shared_ptr<Solver<Real>> m_solver;
    std::vector<std::shared_ptr<Connector>> m_connectors;
};

extern template class SimulationScene<float>;
extern template class SimulationScene<double>;

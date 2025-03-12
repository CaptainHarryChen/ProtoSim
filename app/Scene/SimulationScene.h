#pragma once
#include <vector>
#include <memory>
#include <Object/Object.h>

template <typename Real>
class Solver;
class Connector;

template <typename Real>
class SimulationScene : public Object
{
public:
    SimulationScene() = default;
    virtual ~SimulationScene() = default;

    void Update(double delta_time) override;

    void SetupScene();
    void SetSolver(std::shared_ptr<Solver<Real>> solver);
    std::vector<std::shared_ptr<Connector>> &GetConnectors();

protected:
    std::shared_ptr<Solver<Real>> m_solver;
    std::vector<std::shared_ptr<Connector>> m_connectors;
};

extern template class SimulationScene<float>;
extern template class SimulationScene<double>;

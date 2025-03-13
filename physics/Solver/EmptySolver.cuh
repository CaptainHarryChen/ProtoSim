#pragma once
#include <vector>
#include <Solver/Solver.cuh>

template <typename Real>
struct EmptySolverData
{
    unsigned int num_vertices;
    Real *dev_position;
};

class Mesh;
class Connector;

template <typename Real>
class EmptySolver : public Solver<Real>
{
public:
    EmptySolver(const std::vector<Real> &positions);
    virtual ~EmptySolver();

    virtual void Step() override;
    virtual Real *GetDevicePositions() override;

    EmptySolverData<Real> m_data;

protected:
    EmptySolverData<Real> *m_dev_data;
};

extern template class EmptySolver<float>;
extern template class EmptySolver<double>;

#pragma once

#include <vector>
#include <Solver/Solver.cuh>

enum RigidBodyShape
{
    RIGID_BODY_BOX = 0,
    RIGID_BODY_SPHERE = 1,
    RIGID_BODY_CAPSULE = 2
};

template <typename Real>
struct RigidSolverData
{
    unsigned int num_bodies;

    Real *dev_position;
    Real *dev_orientation;
    Real *dev_linear_velocity;
    Real *dev_angular_velocity;
    Real *dev_mass;
    Real *dev_inv_mass;
    Real *dev_inertia_tensor_local;
    Real *dev_inv_inertia_tensor_local;
    int *dev_shape;
    Real *dev_shape_param;
};

template <typename Real>
class RigidSolver : public Solver<Real>
{
public:
    RigidSolver(
        const std::vector<Real> &position,
        const std::vector<Real> &orientation,
        const std::vector<Real> &linear_velocity,
        const std::vector<Real> &angular_velocity,
        const std::vector<Real> &mass,
        const std::vector<int> &shape,
        const std::vector<Real> &shape_param);
    virtual ~RigidSolver();

    virtual void Step() override;
    virtual Real *GetDevicePositions() override;

    RigidSolverData<Real> m_data;
};

extern template struct RigidSolverData<float>;
extern template struct RigidSolverData<double>;
extern template class RigidSolver<float>;
extern template class RigidSolver<double>;

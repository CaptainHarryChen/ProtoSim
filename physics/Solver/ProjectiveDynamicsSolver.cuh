#pragma once
#include <vector>
#include <Solver/Solver.cuh>

#define MPM_STATIC 0
#define MPM_ELASTIC 1
#define MPM_FLUID 2

const float YOUNG_K = 1000000.0f, YOUNG_NU = 0.26f;
const float LAME_MU = YOUNG_K / (2 * (1 + YOUNG_NU)), LAME_LAMBDA = YOUNG_K * YOUNG_NU / ((1 + YOUNG_NU) * (1 - 2 * YOUNG_NU));
const float GRAVITY = 9.81f;
const float TIME_STEP = 1.0f / 100.0f;

template <typename Real>
struct ProjectiveDynamicsSolverData
{
    unsigned int *dev_vert_to_tet;
    unsigned int *dev_vert_to_tet_offsets;
    Real *dev_diag_Hessian;

    Real *dev_position;
    Real *dev_position_next;
    Real *dev_position_backup;
    Real *dev_init_A;
    Real *dev_init_B;

    Real *dev_tet_force;
    Real *dev_vert_force;
    Real *dev_vert_Hessian;

    Real *dev_energy;

    unsigned int *dev_constrained;
    Real *dev_position_delta;
    Real *dev_inertia;

    unsigned int *dev_ground_collision_count;
    unsigned int *dev_ground_collision_ids;
    unsigned int m_ground_collision_count;

    Real m_time_step;
    Real m_lame_mu;
    Real m_lame_lambda;
    Real m_stiffness = 18000000 * 0.5;
    Real m_collision_stiffness = 1000000;
    Real *dev_gravity;
};

template <typename Real>
class ProjectiveDynamicsSolver : public Solver<Real>
{
public:
    ProjectiveDynamicsSolver();
    virtual ~ProjectiveDynamicsSolver();

    virtual void Step() override;
    virtual Real *GetDevicePositions() override;

    ProjectiveDynamicsSolverData<Real> m_data;

protected:
    ProjectiveDynamicsSolverData<Real> *m_dev_data;
};

extern template struct ProjectiveDynamicsSolverData<float>;
extern template struct ProjectiveDynamicsSolverData<double>;
extern template class ProjectiveDynamicsSolver<float>;
extern template class ProjectiveDynamicsSolver<double>;

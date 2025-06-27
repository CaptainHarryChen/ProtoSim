#pragma once
#include <vector>
#include <Solver/Solver.cuh>

const float YOUNG_K = 1000000.0f, YOUNG_NU = 0.26f;
const float LAME_MU = YOUNG_K / (2 * (1 + YOUNG_NU)), LAME_LAMBDA = YOUNG_K * YOUNG_NU / ((1 + YOUNG_NU) * (1 - 2 * YOUNG_NU));
const float GROUND_COLLISION_STIFFNESS = 1000000.0f;
const float GRAVITY = 9.81f;
const float TIME_STEP = 1.0f / 100.0f;
const unsigned int MAX_ITERATIONS = 150;
const unsigned int CHEBYSHEV_DELAY_ITER = 10;
const float CHEBYSHEV_RHO = 0.9f;
const float UNDER_RELAXATION = 0.7f;

template <typename Real>
struct ProjectiveDynamicsSolverData
{
    unsigned int m_num_vert;
    unsigned int m_num_tet;

    Real *dev_position_backup;
    Real *dev_position_guess;
    Real *dev_position_prev;
    Real *dev_position;
    Real *dev_position_next;
    Real *dev_position_delta;
    Real *dev_velocity;
    Real *dev_mass;
    Real *dev_vert_force;
    Real *dev_constraint_Hessian_diag;
    Real *dev_stiffness_matrix_diag;

    unsigned int *dev_tetrahedron;
    Real *dev_tet_density;
    Real *dev_tet_volume;
    Real *dev_tet_force;
    Real *dev_invDm;

    Real m_time_step;
    Real m_time_step_inv;
    Real m_lame_mu;
    Real m_lame_lambda;
    Real m_ground_collision_stiffness;
    Real m_under_relaxation;
    Real *dev_gravity;
};

template <typename Real>
class ProjectiveDynamicsSolver : public Solver<Real>
{
public:
    ProjectiveDynamicsSolver(const std::vector<Real> &position, const std::vector<unsigned int> &tetrahedron, const std::vector<Real> &tetrahedron_density);
    virtual ~ProjectiveDynamicsSolver();

    virtual void Step() override;
    virtual Real *GetDevicePositions() override;

    ProjectiveDynamicsSolverData<Real> m_data;

protected:
    ProjectiveDynamicsSolverData<Real> *m_dev_data;

    void UpdateChebyshevOmega(Real &omega, unsigned iter);
    void SwapPositionBuffers();
};

extern template struct ProjectiveDynamicsSolverData<float>;
extern template struct ProjectiveDynamicsSolverData<double>;
extern template class ProjectiveDynamicsSolver<float>;
extern template class ProjectiveDynamicsSolver<double>;

#pragma once
#include <vector>
#include <Solver/Solver.cuh>

#define MPM_STATIC 0
#define MPM_ELASTIC 1
#define MPM_FLUID 2

const float YOUNG_K = 1000000.0f;
const float GRAVITY = 9.81f;
const float TIME_STEP = 1.0f / 100.0f;
const unsigned int MAX_ITERATIONS = 150;
const unsigned int CHEBYSHEV_DELAY_ITER = 10;
const float CHEBYSHEV_RHO = 0.9f;
const float UNDER_RELAXATION = 0.7f;

#define ALPHA_MIN 1e-6
#define BETA_LS 0.5

template <typename Real>
struct ProjectiveDynamicsSolverData
{
    unsigned int m_num_vert;
    unsigned int m_num_tet;

    Real *dev_position;
    Real *dev_position_prev;
    Real *dev_position_next;
    Real *dev_position_backup;
    Real *dev_position_delta;
    Real *dev_velocity;
    Real *dev_mass;
    Real *dev_mass_inv;
    Real *dev_inertia;
    Real *dev_init_A;
    Real *dev_init_B;
    Real *dev_vert_force;
    Real *dev_vert_Hessian;
    Real *dev_diag_Hessian;

    unsigned int *dev_vert_to_tet;
    unsigned int *dev_vert_to_tet_offset;

    unsigned int *dev_tetrahedron;
    Real *dev_tet_density;
    Real *dev_tet_volume;
    Real *dev_tet_force;
    Real *dev_invDm;

    Real *dev_energy;

    unsigned int *dev_ground_collision_count;
    unsigned int *dev_ground_collision_ids;

    Real m_time_step;
    Real m_time_step_inv;
    Real m_stiffness;
    Real m_under_relaxation;
    Real *dev_gravity;
};

template <typename Real>
class ProjectiveDynamicsSolver : public Solver<Real>
{
public:
    ProjectiveDynamicsSolver(const std::vector<Real> &position, const std::vector<unsigned int> &tetrahedron, const std::vector<Real> &tetrahedron_density,
                             const std::vector<unsigned int> &object_tetrahedron_offset);
    virtual ~ProjectiveDynamicsSolver();

    virtual void Step() override;
    virtual Real *GetDevicePositions() override;

    ProjectiveDynamicsSolverData<Real> m_data;

protected:
    ProjectiveDynamicsSolverData<Real> *m_dev_data;

    Real line_searches();
    Real compute_energy(Real alpha = 0);
    void UpdateChebysevOmega(Real &omega, unsigned iter);
};

extern template struct ProjectiveDynamicsSolverData<float>;
extern template struct ProjectiveDynamicsSolverData<double>;
extern template class ProjectiveDynamicsSolver<float>;
extern template class ProjectiveDynamicsSolver<double>;

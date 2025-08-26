#pragma once
#include <vector>
#include <Solver/Solver.cuh>

#define MPM_STATIC 0
#define MPM_ELASTIC 1
#define MPM_FLUID 2

const float YOUNG_K = 1000000.0f, YOUNG_NU = 0.26f;
const float LAME_MU = YOUNG_K / (2 * (1 + YOUNG_NU)), LAME_LAMBDA = YOUNG_K * YOUNG_NU / ((1 + YOUNG_NU) * (1 - 2 * YOUNG_NU));
const float GROUND_COLLISION_STIFFNESS = 1000000.0f;
const float GRAVITY = 9.81f;
const float TIME_STEP = 1.0f / 1000.0f;
const unsigned int MAX_ITERATIONS = 150;
const unsigned int CHEBYSHEV_DELAY_ITER = 10;
const float CHEBYSHEV_RHO = 0.9f;
const float UNDER_RELAXATION = 0.7f;

template <typename Real>
struct ImplicitMPMSolverData
{
    unsigned int m_num_particle;
    Real *dev_particle_position;
    Real *dev_particle_velocity;
    Real *dev_particle_mass;
    Real *dev_particle_volume;
    unsigned int *dev_particle_type; // 0: static, 1: elastic, 2: fluid
    Real *dev_particle_C;
    Real *dev_particle_F;                  // deformation gradient
    unsigned int *dev_particle_to_grid_id; // order: x y z

    unsigned int m_num_grid;
    Real m_grid_spacing;
    unsigned int m_boundary_thickness;
    Real *dev_inner_bbox;
    Real *dev_outer_bbox;
    unsigned int *dev_grid_size;
    Real *dev_grid_momentum;
    Real *dev_grid_mass;
    Real *dev_grid_force;
    Real *dev_grid_velocity;
    Real *dev_grid_velocity_hat;
    Real *dev_grid_velocity_prev;
    Real *dev_grid_velocity_next;
    Real *dev_grid_diag_K;
    Real *dev_grid_diag_B;

    Real m_time_step;
    Real m_under_relaxation;
    Real m_ground_stiffness;
    Real m_lame_mu;
    Real m_lame_lambda;
    Real *dev_gravity;
};

template <typename Real>
class ImplicitMPMSolver : public Solver<Real>
{
public:
    ImplicitMPMSolver(const std::vector<Real> &particle_position, const std::vector<unsigned int> &particle_type, const std::vector<Real> &particle_mass, const std::vector<Real> &particle_volume,
              std::vector<Real> bbox, Real grid_spacing, unsigned int boundary_thickness);
    virtual ~ImplicitMPMSolver();

    virtual void Step() override;
    virtual Real *GetDevicePositions() override;

    ImplicitMPMSolverData<Real> m_data;
protected:
    ImplicitMPMSolverData<Real> *m_dev_data;

    void UpdateChebyshevOmega(Real &omega, unsigned iter);
    void SwapAnswerBuffers();
};

extern template struct ImplicitMPMSolverData<float>;
extern template struct ImplicitMPMSolverData<double>;
extern template class ImplicitMPMSolver<float>;
extern template class ImplicitMPMSolver<double>;

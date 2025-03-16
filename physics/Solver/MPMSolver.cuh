#pragma once
#include <vector>
#include <Solver/Solver.cuh>

#define MPM_STATIC 0
#define MPM_ELASTIC 1
#define MPM_FLUID 2

const float YOUNG_K = 1000000.0f, YOUNG_NU = 0.26f;
const float LAME_MU = YOUNG_K / (2 * (1 + YOUNG_NU)), LAME_LAMBDA = YOUNG_K * YOUNG_NU / ((1 + YOUNG_NU) * (1 - 2 * YOUNG_NU));
const float GRAVITY = 9.81f;
const float TIME_STEP = 1.0f / 200.0f;

template <typename Real>
struct MPMSolverData
{
    unsigned int m_num_particle;
    Real *dev_particle_position;
    Real *dev_particle_velocity;
    Real *dev_particle_mass;
    Real *dev_particle_volume;
    unsigned int *dev_particle_type; // 0: static, 1: elastic, 2: fluid
    Real *dev_particle_C;
    Real *dev_particle_affine_momentum;     // become momentum after multiplied by delta x
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
    Real *dev_grid_velocity;

    Real m_time_step;
    Real m_lame_mu;
    Real m_lame_lambda;
    Real *dev_gravity;
};

template <typename Real>
class MPMSolver : public Solver<Real>
{
public:
    MPMSolver(const std::vector<Real> &particle_position, const std::vector<unsigned int> &particle_type, const std::vector<Real> &particle_mass, const std::vector<Real> &particle_volume,
              std::vector<Real> bbox, Real grid_spacing, unsigned int boundary_thickness);
    virtual ~MPMSolver();

    virtual void Step() override;
    virtual Real *GetDevicePositions() override;

    MPMSolverData<Real> m_data;
protected:
    MPMSolverData<Real> *m_dev_data;
};

extern template struct MPMSolverData<float>;
extern template struct MPMSolverData<double>;
extern template class MPMSolver<float>;
extern template class MPMSolver<double>;

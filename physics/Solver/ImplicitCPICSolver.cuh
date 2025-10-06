#pragma once
#include <vector>
#include <Solver/Solver.cuh>

#define MPM_STATIC 0
#define MPM_ELASTIC 1
#define MPM_FLUID 2

const float YOUNG_K = 10000000.0f, YOUNG_NU = 0.26f;
const float LAME_MU = YOUNG_K / (2 * (1 + YOUNG_NU)), LAME_LAMBDA = YOUNG_K * YOUNG_NU / ((1 + YOUNG_NU) * (1 - 2 * YOUNG_NU));
const float GRID_BOX_COLLISION_STIFFNESS = 10000000.0f;
const float SOLID_BOX_COLLISION_STIFFNESS = 10000000.0f;
const float COUPLE_COLLISION_STIFFNESS = 50000.0f;
const float GRAVITY = 9.81f;
const float TIME_STEP = 1.0f / 60.0f;
const unsigned int MAX_ITERATIONS = 200;
const unsigned int CHEBYSHEV_DELAY_ITER = 10;
const float CHEBYSHEV_RHO = 0.9f;
const float UNDER_RELAXATION = 0.7f;

template <typename Real>
struct ImplicitCPICSolverData
{
    unsigned int m_num_particle;

    Real *dev_particle_position;
    Real *dev_particle_velocity;
    Real *dev_particle_mass;
    Real *dev_particle_volume;
    unsigned int *dev_particle_type; // 0: static, 1: elastic, 2: fluid
    Real *dev_particle_C;
    Real *dev_particle_temp_J;
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
    uint64_t *dev_grid_tri_info; // 0~31: (float)closest triangle distance, 32: outside or inside, 33~63: triangle index
    Real *dev_grid_velocity_hat;
    Real *dev_grid_velocity_prev;
    Real *dev_grid_velocity_delta;
    Real *dev_grid_velocity_next;
    Real *dev_grid_diag_Kv;
    Real *dev_grid_diag_B;

    unsigned int m_num_vert;
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

    unsigned int m_num_tet;
    unsigned int *dev_tetrahedron;
    Real *dev_tet_density;
    Real *dev_tet_volume;
    Real *dev_tet_force;
    Real *dev_invDm;

    unsigned int m_num_tri;
    unsigned int *dev_triangle;

    unsigned int m_num_sample;
    Real *dev_sample_position;
    Real *dev_sample_barycentric;
    unsigned int *dev_sample_tri_idx;
    unsigned int *dev_sample_to_grid_id;

    Real m_time_step;
    Real m_time_step_inv;
    Real m_lame_mu;
    Real m_lame_lambda;
    Real m_grid_box_collision_stiffness;
    Real m_solid_box_collision_stiffness;
    Real m_couple_collision_stiffness;
    Real m_under_relaxation;
    Real *dev_gravity;
};

template <typename Real>
class ImplicitCPICSolver : public Solver<Real>
{
public:
    ImplicitCPICSolver(
        const std::vector<Real> &node_position,
        const std::vector<unsigned int> &surface_triangle,
        const std::vector<unsigned int> &tetrahedron,
        const std::vector<Real> &tetrahedron_density,

        const std::vector<Real> &sample_barycentric_weights,
        const std::vector<unsigned int> &sample_triangle_idx,

        const std::vector<Real> &particle_position,
        const std::vector<unsigned int> &particle_type,
        const std::vector<Real> &particle_mass,
        const std::vector<Real> &particle_volume,
        std::vector<Real> bbox,
        Real grid_spacing,
        unsigned int boundary_thickness //
    );
    virtual ~ImplicitCPICSolver();

    virtual void Step() override;
    Real *GetDeviceNodePositions();
    Real *GetDeviceSamplePositions();
    Real *GetDeviceParticlePositions();

    ImplicitCPICSolverData<Real> m_data;
    ImplicitCPICSolverData<Real> *m_dev_data;

protected:

    void UpdateChebyshevOmega(Real &omega, unsigned iter);
    void SwapAnswerBuffers();
};

extern template struct ImplicitCPICSolverData<float>;
extern template struct ImplicitCPICSolverData<double>;
extern template class ImplicitCPICSolver<float>;
extern template class ImplicitCPICSolver<double>;

#pragma once
#include <vector>
#include <Solver/Solver.cuh>

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
struct StrongPDMPMSolverData
{
    unsigned int m_num_vert;
    unsigned int m_num_tet;

    Real *dev_position_backup;
    Real *dev_position_guess;
    Real *dev_position_prev;
    Real *dev_position;
    Real *dev_position_next;
    Real *dev_velocity;
    Real *dev_mass;
    Real *dev_vert_force;
    Real *dev_constraint_Hessian_diag;
    Real *dev_collision_Hessian_diag;
    Real *dev_stiffness_matrix_diag;

    unsigned int *dev_tetrahedron;
    Real *dev_tet_density;
    Real *dev_tet_volume;
    Real *dev_tet_force;
    Real *dev_invDm;

    unsigned int m_num_tri;
    unsigned int *dev_triangle;
    Real *dev_tri_normal;

    unsigned int m_num_sample;
    Real *dev_sample_position;
    Real *dev_sample_barycentric;
    unsigned int *dev_sample_tri_idx;
    Real *dev_sample_volume;
    Real *dev_sample_velocity;
    Real *dev_sample_velocity_backup;
    Real *dev_sample_mass;
    Real *dev_sample_J;
    Real *dev_sample_temp_J;
    Real *dev_sample_C;
    Real *dev_sample_force;
    Real *dev_sample_diag_K;
    unsigned int *dev_sample_to_grid_id;

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
    Real *dev_grid_diag_K;

    Real m_time_step;
    Real m_time_step_inv;
    Real m_lame_mu;
    Real m_lame_lambda;
    Real m_ground_collision_stiffness;
    Real m_under_relaxation;
    Real *dev_gravity;
};

template <typename Real>
class StrongPDMPMSolver : public Solver<Real>
{
public:
    StrongPDMPMSolver(
        const std::vector<Real> &node_position, 
        const std::vector<unsigned int> &surface_triangle,
        const std::vector<unsigned int> &tetrahedron, 
        const std::vector<Real> &tetrahedron_density,

        const std::vector<Real> &sample_barycentric_weights,
        const std::vector<unsigned int> &sample_triangle_idx,
        const std::vector<Real> &sample_volume,

        std::vector<Real> bbox,
        Real grid_spacing,
        unsigned int boundary_thickness
    );
    virtual ~StrongPDMPMSolver();

    virtual void Step() override;
    virtual Real *GetDeviceNodePositions();
    virtual Real *GetDeviceSamplePositions();

    StrongPDMPMSolverData<Real> m_data;

protected:
    StrongPDMPMSolverData<Real> *m_dev_data;

    void UpdateChebyshevOmega(Real &omega, unsigned iter);
    void SwapPositionBuffers();
};

extern template struct StrongPDMPMSolverData<float>;
extern template struct StrongPDMPMSolverData<double>;
extern template class StrongPDMPMSolver<float>;
extern template class StrongPDMPMSolver<double>;

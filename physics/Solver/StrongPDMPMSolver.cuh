#pragma once
#include <vector>
#include <Solver/Solver.cuh>

const float YOUNG_K = 1000000.0f, YOUNG_NU = 0.26f;
const float LAME_MU = YOUNG_K / (2 * (1 + YOUNG_NU)), LAME_LAMBDA = YOUNG_K * YOUNG_NU / ((1 + YOUNG_NU) * (1 - 2 * YOUNG_NU));
const float GROUND_COLLISION_STIFFNESS = 1000000.0f;
const float GRAVITY = 9.81f;
const float TIME_STEP = 1.0f / 200.0f;
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
    Real *dev_stiffness_matrix_diag;

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
        const std::vector<unsigned int> &sample_triangle_idx
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

#pragma once
#include <vector>
#include <Solver/Solver.cuh>

// #define NEWHOOKEAN_MODEL
#define COROTATED_LINEAR_MODEL

#if (defined(NEWHOOKEAN_MODEL) && defined(COROTATED_LINEAR_MODEL)) || (!defined(NEWHOOKEAN_MODEL) && !defined(COROTATED_LINEAR_MODEL))
#error "Only one constitutive model can be defined at a time."
#endif

const float YOUNG_K = 1000000.0f, YOUNG_NU = 0.26f;
const float LAME_MU = YOUNG_K / (2 * (1 + YOUNG_NU)), LAME_LAMBDA = YOUNG_K * YOUNG_NU / ((1 + YOUNG_NU) * (1 - 2 * YOUNG_NU));
const float GROUND_COLLISION_STIFFNESS = 10000000.0f;
const float GRAVITY = 9.81f;
const float TIME_STEP = 1.0f / 60.0f; // Corotated Linear
// const float TIME_STEP = 1.0f / 1000.0f; // Neohookean

// PCG parameters
const unsigned int MAX_ITERATIONS = 20; // for Corotated Linear
// const unsigned int MAX_ITERATIONS = 1000; // for Neohookean
const float RESIDUAL_TOLERANCE = 1e-2f;

template <typename Real>
struct PCGFEMSolverData
{
    unsigned int m_num_vert;
    unsigned int m_num_tet;

    Real *dev_vert_position;
    Real *dev_vert_position_next;
    Real *dev_vert_velocity;
    Real *dev_vert_velocity_hat;
    Real *dev_vert_mass;
    Real *dev_vert_force;
    Real *dev_vert_diag_B;
    Real *dev_vert_box_collision_A;
    Real *dev_vert_p;
    Real *dev_vert_temp; // used for various temporary storage (beta, alpha, Ap, etc.)

    unsigned int *dev_tetrahedron;
    Real *dev_tet_density;
    Real *dev_tet_volume;
    Real *dev_invDm;

    Real m_time_step;
    Real m_time_step_inv;
    Real m_lame_mu;
    Real m_lame_lambda;
    Real m_ground_collision_stiffness;
    Real *dev_gravity;
};

template <typename Real>
class PCGFEMSolver : public Solver<Real>
{
public:
    PCGFEMSolver(const std::vector<Real> &position, const std::vector<unsigned int> &tetrahedron, const std::vector<Real> &tetrahedron_density);
    virtual ~PCGFEMSolver();

    virtual void Step() override;
    virtual Real *GetDevicePositions() override;

    bool m_verbose = false;
    PCGFEMSolverData<Real> m_data;

protected:
    PCGFEMSolverData<Real> *m_dev_data;
};

extern template struct PCGFEMSolverData<float>;
extern template struct PCGFEMSolverData<double>;
extern template class PCGFEMSolver<float>;
extern template class PCGFEMSolver<double>;

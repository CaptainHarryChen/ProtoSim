#pragma once
#include <any>
#include <string>
#include <unordered_map>
#include <vector>
#include <Solver/Solver.cuh>

// #define NEWHOOKEAN_MODEL
#define COROTATED_LINEAR_MODEL

#if (defined(NEWHOOKEAN_MODEL) && defined(COROTATED_LINEAR_MODEL)) || (!defined(NEWHOOKEAN_MODEL) && !defined(COROTATED_LINEAR_MODEL))
#error "Only one constitutive model can be defined at a time."
#endif

template <typename Real>
struct PCGFEMSolverData
{
    unsigned int m_num_vert;
    unsigned int m_num_tet;

    Real *dev_vert_position_prev;
    Real *dev_vert_position;
    Real *dev_vert_velocity_prev;
    Real *dev_vert_velocity;
    Real *dev_vert_velocity_hat;
    Real *dev_vert_mass;
    Real *dev_vert_force;
    Real *dev_vert_diag_B;
    Real *dev_vert_box_collision_A;
    Real *dev_vert_p;
    Real *dev_vert_pAp;
    Real *dev_vert_temp3; // used for various temporary storage (beta, alpha, Ap, etc.)

    unsigned int *dev_tetrahedron;
    Real *dev_tet_density;
    Real *dev_tet_volume;
    Real *dev_invDm;

    Real m_time_step;
    Real m_time_step_inv;
    Real m_lame_mu;
    Real m_lame_lambda;
    Real m_ground_collision_stiffness;
    Real m_gravity[3];
};

template <typename Real>
class PCGFEMSolver : public Solver<Real>
{
public:
    PCGFEMSolver(
        const std::vector<Real> &position,
        const std::vector<unsigned int> &tetrahedron,
        const std::vector<Real> &tetrahedron_density,
        const std::unordered_map<std::string, std::any> &config
    );
    virtual ~PCGFEMSolver();

    virtual void Step() override;
    virtual Real *GetDevicePositions() override;

    void PCG_Preparation();
    void ElasticForceAndPreconditioner();
    void GroundConstraintForceAndPreconditioner();
    Real ResidualNorm();
    Real Calculate_z_dot_r();
    void SearchDirection(Real beta);
    void NormalizeSearchDirection();
    void Calc_pAp_Elastic();
    void Calc_pAp_GroundConstraint();
    Real Calculate_pAp();
    Real Calculate_p_dot_r();
    void UpdateSolution(Real alpha);
    void PCG_After();

    bool m_verbose = false;
    unsigned int m_pcg_max_iteration;
    unsigned int m_line_search_max_iteration;
    Real m_pcg_residual_tolerance;
    PCGFEMSolverData<Real> m_data;
};

extern template struct PCGFEMSolverData<float>;
extern template struct PCGFEMSolverData<double>;
extern template class PCGFEMSolver<float>;
extern template class PCGFEMSolver<double>;

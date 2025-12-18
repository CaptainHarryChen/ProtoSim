#pragma once
#include <any>
#include <string>
#include <unordered_map>
#include <vector>
#include <Solver/Solver.cuh>
#include "PICVolumeCorrector.cuh"

#define MPM_STATIC 0
#define MPM_ELASTIC 1
#define MPM_FLUID 2

template <typename Real>
struct PCGMPMSolverData
{
    unsigned int m_num_particle;
    Real *dev_particle_position;
    Real *dev_particle_velocity;
    Real *dev_particle_mass;
    Real *dev_particle_volume;
    unsigned int *dev_particle_type; // 0: static, 1: elastic, 2: fluid
    Real *dev_particle_C;
    Real *dev_particle_temp_C;
    Real *dev_particle_F;                  // deformation gradient
    unsigned int *dev_particle_to_grid_id; // order: x y z
    Real *dev_particle_temp1; // for temporary G2P when calculate Ap in PCG

    unsigned int m_num_grid;
    Real m_grid_spacing;
    unsigned int m_boundary_thickness;
    Real *dev_inner_bbox;
    Real *dev_outer_bbox;
    unsigned int *dev_grid_size;
    Real *dev_grid_momentum;
    Real *dev_grid_mass;
    Real *dev_grid_force;
    Real *dev_grid_velocity_prev;
    Real *dev_grid_velocity;
    Real *dev_grid_velocity_hat;
    Real *dev_grid_diag_B_const;
    Real *dev_grid_diag_B;
    Real *dev_grid_diag_A_box;
    Real *dev_grid_p; // the search direction in PCG
    Real *dev_grid_pAp;
    Real *dev_grid_temp3; // used for various temporary storage(beta, alpha, Ap, etc.)

    Real m_time_step;
    Real m_time_step_inv;
    Real m_ground_stiffness;
    Real m_fluid_lambda;
    Real m_fluid_viscosity;
    Real *dev_gravity;
};

template <typename Real>
class PCGMPMSolver : public Solver<Real>
{
public:
    PCGMPMSolver(
        const std::vector<Real> &particle_position,
        const std::vector<unsigned int> &particle_type,
        const std::vector<Real> &particle_mass,
        const std::vector<Real> &particle_volume,
        std::vector<Real> bbox,
        Real grid_spacing,
        unsigned int boundary_thickness,
        const std::unordered_map<std::string, std::any> &config
    );
    virtual ~PCGMPMSolver();

    virtual void Step() override;
    virtual Real *GetDevicePositions() override;

    void PCG_Preparation();
    void MaterialForceAndPreconditioner();
    void BoxConstraintForceAndPreconditioner();
    Real ResidualNorm();
    Real Calculate_z_dot_r();
    void SearchDirection(Real beta);
    void NormalizeSearchDirection();
    void Calc_pAp_Material();
    void Calc_pAp_BoxConstraint();
    Real Calculate_pAp();
    Real Calculate_p_dot_r();
    void UpdateSolution(Real alpha);
    void PCG_After();

    bool m_verbose = false;
    unsigned int m_pcg_max_iteration;
    unsigned int m_line_search_max_iteration;
    Real m_pcg_residual_tolerance;
    PCGMPMSolverData<Real> m_data;
    PCGMPMSolverData<Real> *m_dev_data;

    PICVolumeCorrector<Real> *m_volume_corrector = nullptr;
};

extern template struct PCGMPMSolverData<float>;
extern template struct PCGMPMSolverData<double>;
extern template class PCGMPMSolver<float>;
extern template class PCGMPMSolver<double>;

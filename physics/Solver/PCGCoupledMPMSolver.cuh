#pragma once
#include <Solver/Solver.cuh>
#include "PCGMPMSolver.cuh"
#include "PCGFEMSolver.cuh"

// #define SEPARATE_SOLVE

template <typename Real>
struct PCGCoupledMPMSolverData
{
    const PCGFEMSolverData<Real> *dev_fem_data;
    const PCGMPMSolverData<Real> *dev_mpm_data;

    // surface triangle mesh for coupling
    unsigned int m_num_triangle;
    unsigned int *dev_triangle;
    // sampled particles on the surface mesh for coupling
    unsigned int m_num_sample;
    Real *dev_sample_position;
    Real *dev_sample_barycentric;
    Real *dev_sample_area;
    unsigned int *dev_sample_tri_idx;
    unsigned int *dev_sample_to_grid_id;
    Real *dev_sample_temp3;

    Real *dev_particle_color;
    Real *dev_particle_temp_position; // for coupling force
    unsigned int *dev_temp_particle_to_grid_id; // for coupling force
    Real *dev_particle_dis;
    Real *dev_particle_normal;
    Real *dev_particle_temp3; // for coupling force Ap calculation

    uint64_t *dev_grid_tri_info; // for contact constraint
    Real *dev_grid_nnT; // 3x3 matrix, for contact constraint: sum_p w_ip * n_p * n_p^T
    Real *dev_grid_contact_force; // store contact force
    Real *dev_grid_sample_area;

    Real m_time_step;
    Real m_time_step_inv;
    Real m_contact_stiffness;
};

template <typename Real>
class PCGCoupledMPMSolver : public Solver<Real>
{
public:
    PCGCoupledMPMSolver(
        const std::vector<Real> &vert_position,
        const std::vector<unsigned int> &surface_triangle,
        const std::vector<unsigned int> &tetrahedron,
        const std::vector<Real> &tetrahedron_density,

        const std::vector<Real> &sample_barycentric_weights,
        const std::vector<unsigned int> &sample_triangle_idx,
        const std::vector<Real> &sample_area,

        const std::vector<Real> &particle_position,
        const std::vector<unsigned int> &particle_type,
        const std::vector<Real> &particle_mass,
        const std::vector<Real> &particle_volume,
        std::vector<Real> bbox,
        Real grid_spacing,
        unsigned int boundary_thickness,
        const std::unordered_map<std::string, std::any> &config
    );
    virtual ~PCGCoupledMPMSolver();

    virtual void Step() override;
    Real *GetDeviceVertexPositions();
    Real *GetDeviceSamplePositions();
    Real *GetDeviceParticlePositions();

    void BoxConstraintForceAndPreconditioner();
    void ContactConstraintForceAndPreconditioner();
    void NormalizeSearchDirection();
    void Calc_pAp_BoxConstraint();
    void Calc_pAp_ContactConstraint();

    bool m_verbose = false;
    PCGCoupledMPMSolverData<Real> m_data;
    PCGCoupledMPMSolverData<Real> *m_dev_data;
    PCGMPMSolver<Real> m_mpm_solver;
    PCGFEMSolver<Real> m_fem_solver;
};

extern template struct PCGCoupledMPMSolverData<float>;
extern template struct PCGCoupledMPMSolverData<double>;
extern template class PCGCoupledMPMSolver<float>;
extern template class PCGCoupledMPMSolver<double>;

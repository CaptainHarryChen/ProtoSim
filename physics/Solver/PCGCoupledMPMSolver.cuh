#pragma once
#include <Solver/Solver.cuh>
#include "PCGMPMSolver.cuh"
#include "PCGFEMSolver.cuh"

template <typename Real>
struct PCGCoupledMPMSolverData
{
    Real *dev_fem_vert_position;

    Real *dev_mpm_particle_position;

    unsigned int m_num_triangle;
    unsigned int *dev_triangle;

    unsigned int m_num_sample;
    Real *dev_sample_position;
    Real *dev_sample_barycentric;
    unsigned int *dev_sample_tri_idx;
    unsigned int *dev_sample_to_grid_id;

    Real m_time_step;
    Real m_time_step_inv;
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

    PCGCoupledMPMSolverData<Real> m_data;
    bool m_verbose = false;
protected:
    PCGCoupledMPMSolverData<Real> *m_dev_data;
    PCGMPMSolver<Real> m_mpm_solver;
    PCGFEMSolver<Real> m_fem_solver;
};

extern template struct PCGCoupledMPMSolverData<float>;
extern template struct PCGCoupledMPMSolverData<double>;
extern template class PCGCoupledMPMSolver<float>;
extern template class PCGCoupledMPMSolver<double>;

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
const float COUPLE_COLLISION_STIFFNESS = 10000.0f;
const float GRAVITY = 9.81f;

const float FLUID_BULK_MODULUS = 1000000.0f;
const float FLUID_VISCOSITY_COEFF = 10.0f;

const float TIME_STEP = 1.0f / 200.0f;
const unsigned int FEM_PCG_MAX_ITERATIONS = 20;
const float FEM_RESIDUAL_TOLERANCE = 1e-2f;
const unsigned int MPM_PCG_MAX_ITERATIONS = 100000;
const float MPM_RESIDUAL_TOLERANCE = 1e-2f;


template <typename Real>
struct PCGCoupledMPMSolverData
{
    // FEM data --------------------------------------------------
    unsigned int fem_node_count = 0;
    unsigned int fem_tetrahedron_count = 0;

    Real *dev_fem_node_position = nullptr;
    Real *dev_fem_node_position_next = nullptr;
    Real *dev_fem_node_velocity = nullptr;
    Real *dev_fem_node_velocity_hat = nullptr;
    Real *dev_fem_node_mass = nullptr;
    Real *dev_fem_node_force = nullptr;
    Real *dev_fem_node_diag = nullptr;
    Real *dev_fem_node_collision_diag = nullptr;
    Real *dev_fem_search_direction = nullptr;
    Real *dev_fem_scratch = nullptr;

    unsigned int *dev_fem_tetrahedron = nullptr;
    Real *dev_fem_tet_density = nullptr;
    Real *dev_fem_tet_volume = nullptr;
    Real *dev_fem_inv_dm = nullptr;

    // Surface / sample data -------------------------------------
    unsigned int surface_triangle_count = 0;
    unsigned int *dev_surface_triangles = nullptr;

    unsigned int sample_count = 0;
    Real *dev_sample_position = nullptr;
    Real *dev_sample_barycentric = nullptr;
    unsigned int *dev_sample_triangle_index = nullptr;

    // MPM particle data ----------------------------------------
    unsigned int mpm_particle_count = 0;
    Real *dev_mpm_particle_position = nullptr;
    Real *dev_mpm_particle_velocity = nullptr;
    Real *dev_mpm_particle_mass = nullptr;
    Real *dev_mpm_particle_volume = nullptr;
    unsigned int *dev_mpm_particle_type = nullptr;
    Real *dev_mpm_particle_C = nullptr;
    Real *dev_mpm_particle_temp_C = nullptr;
    Real *dev_mpm_particle_F = nullptr;
    unsigned int *dev_mpm_particle_cell = nullptr;
    Real *dev_mpm_particle_scalar = nullptr;

    // MPM grid data --------------------------------------------
    unsigned int mpm_grid_count = 0;
    Real mpm_grid_spacing = 0;
    unsigned int mpm_boundary_thickness = 0;
    Real *dev_mpm_inner_bbox = nullptr;
    Real *dev_mpm_outer_bbox = nullptr;
    unsigned int *dev_mpm_grid_resolution = nullptr;
    Real *dev_mpm_grid_momentum = nullptr;
    Real *dev_mpm_grid_mass = nullptr;
    Real *dev_mpm_grid_force = nullptr;
    Real *dev_mpm_grid_velocity = nullptr;
    Real *dev_mpm_grid_velocity_hat = nullptr;
    Real *dev_mpm_grid_diag_const = nullptr;
    Real *dev_mpm_grid_diag_mutable = nullptr;
    Real *dev_mpm_grid_search = nullptr;
    Real *dev_mpm_grid_temp = nullptr;

    // Shared simulation constants -------------------------------
    Real fem_time_step = (Real)TIME_STEP;
    Real fem_time_step_inv = (Real)(1.0f / TIME_STEP);
    Real fem_lame_mu = (Real)LAME_MU;
    Real fem_lame_lambda = (Real)LAME_LAMBDA;
    Real fem_collision_stiffness = (Real)SOLID_BOX_COLLISION_STIFFNESS;

    Real mpm_time_step = (Real)TIME_STEP;
    Real mpm_ground_stiffness = (Real)GRID_BOX_COLLISION_STIFFNESS;
    Real mpm_fluid_lambda = (Real)FLUID_BULK_MODULUS;
    Real mpm_fluid_viscosity = (Real)FLUID_VISCOSITY_COEFF;

    Real *dev_gravity = nullptr;
};

template <typename Real>
class PCGCoupledMPMSolver : public Solver<Real>
{
public:
    PCGCoupledMPMSolver(
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
    virtual ~PCGCoupledMPMSolver();

    virtual void Step() override;
    Real *GetDeviceNodePositions();
    Real *GetDeviceSamplePositions();
    Real *GetDeviceParticlePositions();

    PCGCoupledMPMSolverData<Real> m_data;
    PCGCoupledMPMSolverData<Real> *m_dev_data;
    bool m_verbose = false;
};

extern template struct PCGCoupledMPMSolverData<float>;
extern template struct PCGCoupledMPMSolverData<double>;
extern template class PCGCoupledMPMSolver<float>;
extern template class PCGCoupledMPMSolver<double>;

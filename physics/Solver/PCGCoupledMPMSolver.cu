#include "PCGCoupledMPMSolver.cuh"
#include <cuda_utils/cuda_utils.cuh>
#include <Math/algebra.cuh>

namespace CoupledMPMSolverKernel
{
    template <typename Real>
    __global__ void update_sample_position(PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int s = blockDim.x * blockIdx.x + threadIdx.x;
        if (s >= data->m_num_sample)
            return;
        unsigned int tri_idx = (unsigned int)data->dev_sample_tri_idx[s];
        Real pos[3] = {0, 0, 0};
        for (unsigned int v = 0; v < 3; v++)
        {
            unsigned int vert_idx = data->dev_triangle[tri_idx * 3 + v];
            Real weight = data->dev_sample_barycentric[s * 3 + v];
            cudaPhysics::axpby(pos, (Real)1, pos, weight, &data->dev_fem_vert_position[vert_idx * 3], 3);
        }
        cudaPhysics::vecCopy(&data->dev_sample_position[s * 3], pos, 3);
    }
}

template <typename Real>
PCGCoupledMPMSolver<Real>::PCGCoupledMPMSolver(
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
) :
        m_mpm_solver(
            particle_position,
            particle_type,
            particle_mass,
            particle_volume,
            bbox,
            grid_spacing,
            boundary_thickness,
            config),
        m_fem_solver(
            vert_position,
            tetrahedron,
            tetrahedron_density,
            config)
{
    m_data.dev_fem_vert_position = m_fem_solver.m_data.dev_vert_position;
    m_data.dev_mpm_particle_position = m_mpm_solver.m_data.dev_particle_position;

    m_data.m_num_triangle = static_cast<unsigned int>(surface_triangle.size() / 3);
    m_data.m_num_sample = static_cast<unsigned int>(sample_triangle_idx.size());
    printf("PCGCoupledMPMSolver: num_triangle: %u, num_sample: %u\n", m_data.m_num_triangle, m_data.m_num_sample);

    cudaMalloc((void **)&m_data.dev_triangle, sizeof(unsigned int) * m_data.m_num_triangle * 3);
    cudaMemcpy(m_data.dev_triangle, surface_triangle.data(), sizeof(unsigned int) * m_data.m_num_triangle * 3, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&m_data.dev_sample_position, sizeof(Real) * m_data.m_num_sample * 3);
    cudaMalloc((void **)&m_data.dev_sample_barycentric, sizeof(Real) * m_data.m_num_sample * 3);
    cudaMemcpy(m_data.dev_sample_barycentric, sample_barycentric_weights.data(), sizeof(Real) * m_data.m_num_sample * 3, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_sample_tri_idx, sizeof(unsigned int) * m_data.m_num_sample);
    cudaMemcpy(m_data.dev_sample_tri_idx, sample_triangle_idx.data(), sizeof(unsigned int) * m_data.m_num_sample, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_sample_to_grid_id, sizeof(unsigned int) * m_data.m_num_sample);

    cudaMalloc(&m_dev_data, sizeof(PCGCoupledMPMSolverData<Real>));
    cudaMemcpy(m_dev_data, &m_data, sizeof(PCGCoupledMPMSolverData<Real>), cudaMemcpyHostToDevice);

    CoupledMPMSolverKernel::update_sample_position<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaCheck(cudaDeviceSynchronize());
    printf("PCGCoupledMPMSolver initialized.\n");
}

template <typename Real>
PCGCoupledMPMSolver<Real>::~PCGCoupledMPMSolver()
{
    cudaFree(m_data.dev_triangle);
    cudaFree(m_data.dev_sample_position);
    cudaFree(m_data.dev_sample_barycentric);
    cudaFree(m_data.dev_sample_tri_idx);
    cudaFree(m_data.dev_sample_to_grid_id);
}

template <typename Real>
void PCGCoupledMPMSolver<Real>::Step()
{
    m_fem_solver.PCG_Preparation();
    m_mpm_solver.PCG_Preparation();
    Real fem_prev_z_dot_r = 0;
    Real mpm_prev_z_dot_r = 0;
    for (unsigned int iter = 0;; ++iter)
    {
        if (m_verbose)
            printf("PCG iteration %u\n", iter);

        cudaMemset(m_fem_solver.m_data.dev_vert_force, 0, sizeof(Real) * m_fem_solver.m_data.m_num_vert * 3);
        cudaMemset(m_fem_solver.m_data.dev_vert_diag_B, 0, sizeof(Real) * m_fem_solver.m_data.m_num_vert * 3);
        cudaMemset(m_mpm_solver.m_data.dev_grid_force, 0, sizeof(Real) * m_mpm_solver.m_data.m_num_grid * 3);
        cudaMemcpy(m_mpm_solver.m_data.dev_grid_diag_B, m_mpm_solver.m_data.dev_grid_diag_B_const, sizeof(Real) * m_mpm_solver.m_data.m_num_grid * 3, cudaMemcpyDeviceToDevice);
        m_fem_solver.ElasticForceAndPreconditioner();
        m_fem_solver.GroundConstraintForceAndPreconditioner();
        m_mpm_solver.MaterialForceAndPreconditioner();
        m_mpm_solver.BoxConstraintForceAndPreconditioner();
        
        Real fem_residual = m_fem_solver.ResidualNorm();
        assert(!std::isnan(fem_residual));
        Real mpm_residual = m_mpm_solver.ResidualNorm();
        assert(!std::isnan(mpm_residual));
        if (m_verbose)
            printf("  fem_residual = %e, mpm_residual = %e\n", fem_residual, mpm_residual);
        if ((fem_residual < m_fem_solver.m_pcg_residual_tolerance || iter >= m_fem_solver.m_pcg_max_iteration )
            && (mpm_residual < m_mpm_solver.m_pcg_residual_tolerance || iter >= m_mpm_solver.m_pcg_max_iteration))
            break;
        m_fem_solver.SearchDirection(fem_prev_z_dot_r);
        m_fem_solver.NormalizeSearchDirection();
        m_mpm_solver.SearchDirection(mpm_prev_z_dot_r);
        m_mpm_solver.NormalizeSearchDirection();
    
        cudaMemset(m_fem_solver.m_data.dev_vert_pAp, 0, sizeof(Real) * m_fem_solver.m_data.m_num_vert * 3);
        cudaMemset(m_mpm_solver.m_data.dev_grid_pAp, 0, sizeof(Real) * m_mpm_solver.m_data.m_num_grid * 3);
        m_fem_solver.Calc_pAp_Elastic();
        m_fem_solver.Calc_pAp_GroundConstraint();
        m_mpm_solver.Calc_pAp_Material();
        m_mpm_solver.Calc_pAp_BoxConstraint();

        m_fem_solver.UpdateSolution();
        m_mpm_solver.UpdateSolution();
    }
    m_fem_solver.PCG_After();
    m_mpm_solver.PCG_After();

    CoupledMPMSolverKernel::update_sample_position<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
}

template <typename Real>
Real *PCGCoupledMPMSolver<Real>::GetDeviceVertexPositions()
{
    return m_data.dev_fem_vert_position;
}

template <typename Real>
Real *PCGCoupledMPMSolver<Real>::GetDeviceSamplePositions()
{
    return m_data.dev_sample_position;
}

template <typename Real>
Real *PCGCoupledMPMSolver<Real>::GetDeviceParticlePositions()
{
    return m_data.dev_mpm_particle_position;
}

template class PCGCoupledMPMSolver<float>;
template class PCGCoupledMPMSolver<double>;
template struct PCGCoupledMPMSolverData<float>;
template struct PCGCoupledMPMSolverData<double>;

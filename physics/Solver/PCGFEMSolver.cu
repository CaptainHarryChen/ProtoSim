#include "PCGFEMSolver.cuh"
#include <cuda_utils/cuda_utils.cuh>
#include <Math/algebra.cuh>
#include <Math/ConstitutiveModel/CorotatedLinear.cuh>
#include <Math/ConstitutiveModel/Neohookean.cuh>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

namespace PCGFEMSolverKernel
{
    template <typename Real>
    __global__ void tetrahedron_initialize(PCGFEMSolverData<Real> data)
    {
        unsigned int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= data.m_num_tet)
            return;
        unsigned int ind[4];
        for (unsigned int i = 0; i < 4; ++i)
            ind[i] = data.dev_tetrahedron[t * 4 + i] * 3;
        Real Dm[9];
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                Dm[i * 3 + j] = data.dev_vert_position[ind[j + 1] + i] - data.dev_vert_position[ind[0] + i];
        Real vol = cudaPhysics::det3(Dm);
        data.dev_tet_volume[t] = abs(vol) / 6.0f;
        cudaPhysics::matInv3(&data.dev_invDm[t * 9], Dm);
        for (unsigned int i = 0; i < 4; ++i)
            atomicAdd(&data.dev_vert_mass[data.dev_tetrahedron[t * 4 + i]], 0.25 * data.dev_tet_volume[t] * data.dev_tet_density[t]);
    }

    template <typename Real>
    __global__ void initial_guess(PCGFEMSolverData<Real> data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data.m_num_vert)
            return;
        cudaPhysics::axpby(&data.dev_vert_p[3 * v], (Real)1.0, &data.dev_vert_velocity[3 * v], data.m_time_step, data.m_gravity, 3);
    }

    template <typename Real>
    __global__ void tet_stiffness_matrix_diag(PCGFEMSolverData<Real> data)
    {
        unsigned int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= data.m_num_tet)
            return;
        
        #ifdef NEWHOOKEAN_MODEL
        unsigned int ind[4];
        for (unsigned int i = 0; i < 4; ++i)
            ind[i] = data.dev_tetrahedron[t * 4 + i] * 3;
        Real K[12];
        cudaPhysics::calc_neohookean_K_diag<Real>(
            K,
            &data.dev_vert_position[ind[0]],
            &data.dev_vert_position[ind[1]],
            &data.dev_vert_position[ind[2]],
            &data.dev_vert_position[ind[3]],
            &data.dev_invDm[t * 9],
            data.dev_tet_volume[t],
            data.m_lame_mu,
            data.m_lame_lambda);
        for (unsigned int i = 0; i < 4; ++i)
        {
            for (unsigned int j = 0; j < 3; ++j)
            {
                // dev_vert_diag_B stores (K = partial f / partial x) instead of B for now
                atomicAdd(&data.dev_vert_diag_B[ind[i] + j], K[i * 3 + j]);
            }
        }
        #endif
        #ifdef COROTATED_LINEAR_MODEL
        Real K[4];
        cudaPhysics::calc_corotated_linear_K_diag<Real>(
            K,
            &data.dev_invDm[t * 9],
            data.dev_tet_volume[t],
            data.m_lame_mu);
        for (unsigned int i = 0; i < 4; ++i)
        {
            for (unsigned int j = 0; j < 3; ++j)
            {
                // dev_vert_diag_B stores (K = partial f / partial x) instead of B for now
                atomicAdd(&data.dev_vert_diag_B[data.dev_tetrahedron[t * 4 + i] * 3 + j], K[i]);
            }
        }
        #endif
    }

    template <typename Real>
    __global__ void calc_vert_diag_B(PCGFEMSolverData<Real> data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data.m_num_vert)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            // A = m / dt - (partial f / partial x) * dt
            data.dev_vert_diag_B[v * 3 + j] = data.dev_vert_mass[v] * data.m_time_step_inv - data.dev_vert_diag_B[v * 3 + j] * data.m_time_step;
        }
    }

    template <typename Real>
    __global__ void calc_tetrahedron_force(PCGFEMSolverData<Real> data)
    {
        unsigned int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= data.m_num_tet)
            return;
        unsigned int ind[4];
        for (unsigned int i = 0; i < 4; ++i)
            ind[i] = data.dev_tetrahedron[t * 4 + i] * 3;
        
        Real Ds[9];
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                Ds[i * 3 + j] = data.dev_vert_position[ind[j + 1] + i] - data.dev_vert_position[ind[0] + i];
        const Real *InvDm = &data.dev_invDm[t * 9];
        Real F[9], P[9], H[9], f0[3] = {0};
        cudaPhysics::matMul3(F, Ds, InvDm);
        #ifdef NEWHOOKEAN_MODEL
        cudaPhysics::calc_neohookean_P<Real>(P, F, data.m_lame_mu, data.m_lame_lambda);
        #endif
        #ifdef COROTATED_LINEAR_MODEL
        cudaPhysics::calc_corotated_linear_P<Real>(P, F, data.m_lame_mu);
        #endif
        cudaPhysics::matmatTMul3(H, P, InvDm);
        cudaPhysics::vecMul(H, -data.dev_tet_volume[t], H, 9);
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                f0[i] -= H[i * 3 + j];
        
        for (unsigned int j = 0; j < 3; ++j)
            atomicAdd(&data.dev_vert_force[ind[0] + j], f0[j]);
        for (unsigned int i = 1; i < 4; ++i)
        {
            for (unsigned int j = 0; j < 3; ++j)
                atomicAdd(&data.dev_vert_force[ind[i] + j], H[j * 3 + i - 1]);
        }
    }

    template <typename Real>
    __global__ void calc_ground_collision_force(PCGFEMSolverData<Real> data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data.m_num_vert)
            return;
        if (data.dev_vert_position[3 * v + 1] < 0.0f)
        {
            data.dev_vert_force[3 * v + 1] += -data.m_ground_collision_stiffness * data.dev_vert_position[3 * v + 1] * data.dev_vert_mass[v];
            data.dev_vert_diag_B[3 * v + 1] += data.m_ground_collision_stiffness * data.dev_vert_mass[v] * data.m_time_step;
            data.dev_vert_box_collision_A[3 * v + 1] += data.m_ground_collision_stiffness * data.dev_vert_mass[v] * data.m_time_step;
        }
    }

    template <typename Real>
    __global__ void vert_r_dot_r(PCGFEMSolverData<Real> data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data.m_num_vert)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            Real r = - (data.dev_vert_mass[v] * data.m_time_step_inv * (data.dev_vert_velocity[3 * v + j] - data.dev_vert_velocity_hat[3 * v + j]) - data.dev_vert_force[3 * v + j]);
            data.dev_vert_temp3[3 * v + j] = r * r;
        }
    }

    template <typename Real>
    __global__ void vert_z_dot_r(PCGFEMSolverData<Real> data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data.m_num_vert)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            Real r = - (data.dev_vert_mass[v] * data.m_time_step_inv * (data.dev_vert_velocity[3 * v + j] - data.dev_vert_velocity_hat[3 * v + j]) - data.dev_vert_force[3 * v + j]);
            data.dev_vert_temp3[3 * v + j] = r * r / data.dev_vert_diag_B[3 * v + j];
        }
    }

    template <typename Real>
    __global__ void vert_search_direction(PCGFEMSolverData<Real> data, Real beta)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data.m_num_vert)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            Real r = - (data.dev_vert_mass[v] * data.m_time_step_inv * (data.dev_vert_velocity[3 * v + j] - data.dev_vert_velocity_hat[3 * v + j]) - data.dev_vert_force[3 * v + j]);
            data.dev_vert_p[v * 3 + j] = r / data.dev_vert_diag_B[3 * v + j]
                                          + beta * data.dev_vert_p[v * 3 + j];
        }
    }

    template <typename Real>
    __global__ void calc_tet_Ap(PCGFEMSolverData<Real> data)
    {
        unsigned int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= data.m_num_tet)
            return;
        unsigned int ind[4];
        for (unsigned int i = 0; i < 4; ++i)
            ind[i] = data.dev_tetrahedron[t * 4 + i] * 3;
        const Real *InvDm = &data.dev_invDm[t * 9];
        Real partial_Ds[9];
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                partial_Ds[i * 3 + j] = (data.dev_vert_p[ind[j + 1] + i] - data.dev_vert_p[ind[0] + i]) * data.m_time_step;
        
        Real Ku[12];
        #ifdef NEWHOOKEAN_MODEL
        Real Ds[9];
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                Ds[i * 3 + j] = data.dev_vert_position[ind[j + 1] + i] - data.dev_vert_position[ind[0] + i];
        cudaPhysics::calc_neohookean_K_mul_u<Real>(
            Ku,
            Ds,
            partial_Ds,
            InvDm,
            data.dev_tet_volume[t],
            data.m_lame_mu,
            data.m_lame_lambda);
        #endif
        #ifdef COROTATED_LINEAR_MODEL
        cudaPhysics::calc_corotated_linear_K_mul_u<Real>(
            Ku,
            partial_Ds,
            InvDm,
            data.dev_tet_volume[t],
            data.m_lame_mu);
        #endif
        for (unsigned int i = 0; i < 4; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                atomicAdd(&data.dev_vert_temp3[ind[i] + j], Ku[i * 3 + j]);
    }
    
    template <typename Real>
    __global__ void calc_elastic_pAp(PCGFEMSolverData<Real> data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data.m_num_vert)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            Real Ap =   data.dev_vert_mass[v] * data.m_time_step_inv * data.dev_vert_p[3 * v + j] 
                      - data.dev_vert_temp3[3 * v + j]; // elastic force part stored in dev_vert_temp3
            data.dev_vert_pAp[3 * v + j] += data.dev_vert_p[3 * v + j] * Ap;
        }
    }

    template <typename Real>
    __global__ void calc_ground_constraint_pAp(PCGFEMSolverData<Real> data)
    {
        unsigned int j = blockDim.x * blockIdx.x + threadIdx.x;
        if (j >= data.m_num_vert * 3)
            return;
        data.dev_vert_pAp[j] += data.dev_vert_box_collision_A[j] * data.dev_vert_p[j] * data.dev_vert_p[j];
    }

    template <typename Real>
    __global__ void vert_p_dot_r(PCGFEMSolverData<Real> data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data.m_num_vert)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            Real r = - (data.dev_vert_mass[v] * data.m_time_step_inv * (data.dev_vert_velocity[3 * v + j] - data.dev_vert_velocity_hat[3 * v + j]) - data.dev_vert_force[3 * v + j]);
            data.dev_vert_temp3[3 * v + j] = data.dev_vert_p[3 * v + j] * r;
        }
    }
}

template <typename Real>
PCGFEMSolver<Real>::PCGFEMSolver(
    const std::vector<Real> &position,
    const std::vector<unsigned int> &tetrahedron,
    const std::vector<Real> &tetrahedron_density,
    const std::unordered_map<std::string, std::any> &config
)
{
    m_data.m_num_vert = (unsigned int)position.size() / 3;
    m_data.m_num_tet = (unsigned int)tetrahedron.size() / 4;

    printf("PCGFEMSolver: num_vert=%u, num_tet=%u\n", m_data.m_num_vert, m_data.m_num_tet);

    cudaMalloc((void **)&m_data.dev_vert_position_prev, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemcpy(m_data.dev_vert_position_prev, position.data(), sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_vert_position, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemcpy(m_data.dev_vert_position, m_data.dev_vert_position_prev, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);
    cudaMalloc((void **)&m_data.dev_vert_velocity_prev, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_vert_velocity, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemset(m_data.dev_vert_velocity, 0, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_vert_velocity_hat, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_vert_mass, sizeof(Real) * m_data.m_num_vert);
    cudaMemset(m_data.dev_vert_mass, 0, sizeof(Real) * m_data.m_num_vert);
    cudaMalloc((void **)&m_data.dev_vert_force, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_vert_diag_B, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_vert_box_collision_A, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_vert_p, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_vert_pAp, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_vert_temp3, sizeof(Real) * m_data.m_num_vert * 3);

    cudaMalloc((void **)&m_data.dev_tetrahedron, sizeof(unsigned int) * tetrahedron.size());
    cudaMemcpy(m_data.dev_tetrahedron, tetrahedron.data(), sizeof(unsigned int) * tetrahedron.size(), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_tet_density, sizeof(Real) * m_data.m_num_tet);
    cudaMemcpy(m_data.dev_tet_density, tetrahedron_density.data(), sizeof(Real) * m_data.m_num_tet, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_tet_volume, sizeof(Real) * m_data.m_num_tet);
    cudaMemset(m_data.dev_tet_volume, 0, sizeof(Real) * m_data.m_num_tet);
    cudaMalloc((void **)&m_data.dev_invDm, sizeof(Real) * m_data.m_num_tet * 9);

    assert(config.find("time_step") != config.end());
    m_data.m_time_step = std::any_cast<Real>(config.at("time_step"));
    m_data.m_time_step_inv = 1.0f / m_data.m_time_step;
    assert(config.find("lame_mu") != config.end());
    m_data.m_lame_mu = std::any_cast<Real>(config.at("lame_mu"));
    assert(config.find("lame_lambda") != config.end());
    m_data.m_lame_lambda = std::any_cast<Real>(config.at("lame_lambda"));
    assert(config.find("ground_collision_stiffness") != config.end());
    m_data.m_ground_collision_stiffness = std::any_cast<Real>(config.at("ground_collision_stiffness"));
    assert(config.find("line_search_max_iteration") != config.end());
    m_line_search_max_iteration = std::any_cast<unsigned int>(config.at("line_search_max_iteration"));
    assert(config.find("fem_pcg_max_iteration") != config.end());
    m_pcg_max_iteration = std::any_cast<unsigned int>(config.at("fem_pcg_max_iteration"));
    assert(config.find("fem_pcg_residual_tolerance") != config.end());
    m_pcg_residual_tolerance = std::any_cast<Real>(config.at("fem_pcg_residual_tolerance"));
    assert(config.find("gravity") != config.end());
    std::vector<Real> gravity = std::any_cast<std::vector<Real>>(config.at("gravity"));
    assert(gravity.size() == 3);
    for (unsigned int i = 0; i < 3; ++i)
        m_data.m_gravity[i] = gravity[i];

    PCGFEMSolverKernel::tetrahedron_initialize<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_data);
    cudaCheck(cudaDeviceSynchronize());
    printf("PCGFEMSolver initialized.\n");
}

template <typename Real>
PCGFEMSolver<Real>::~PCGFEMSolver()
{
    cudaFree(m_data.dev_vert_position_prev);
    cudaFree(m_data.dev_vert_position);
    cudaFree(m_data.dev_vert_velocity_prev);
    cudaFree(m_data.dev_vert_velocity);
    cudaFree(m_data.dev_vert_velocity_hat);
    cudaFree(m_data.dev_vert_mass);
    cudaFree(m_data.dev_vert_force);
    cudaFree(m_data.dev_vert_diag_B);
    cudaFree(m_data.dev_vert_box_collision_A);
    cudaFree(m_data.dev_vert_p);
    cudaFree(m_data.dev_vert_pAp);
    cudaFree(m_data.dev_vert_temp3);

    cudaFree(m_data.dev_tetrahedron);
    cudaFree(m_data.dev_tet_density);
    cudaFree(m_data.dev_tet_volume);
    cudaFree(m_data.dev_invDm);
}

template <typename Real>
void PCGFEMSolver<Real>::Step()
{
    PCG_Preparation();

    Real last_residual = std::numeric_limits<Real>::max();
    Real alpha = 1.0f;
    Real prev_z_dot_r = 0; // for calculate beta in PCG
    for (unsigned int iter = 0;; ++iter)
    {
        if (m_verbose)
            printf("PCG iteration %u\n  last_residual = %e\n", iter, last_residual);
        Real residual = last_residual;
        for (unsigned int ls_iter = 0;;)
        {
            UpdateSolution(alpha);
            cudaMemset(m_data.dev_vert_force, 0, sizeof(Real) * m_data.m_num_vert * 3);
            cudaMemset(m_data.dev_vert_diag_B, 0, sizeof(Real) * m_data.m_num_vert * 3);
            ElasticForceAndPreconditioner();
            GroundConstraintForceAndPreconditioner();

            residual = ResidualNorm();
            if (m_verbose)
                printf("  line_search_iter %u: residual = %e\n", ls_iter, residual);
            ls_iter++;
            if ((residual <= last_residual || ls_iter >= m_line_search_max_iteration) && !std::isnan(residual))
                break;
            alpha *= 0.5f;
        }
        last_residual = residual;
        cudaMemcpy(m_data.dev_vert_velocity_prev, m_data.dev_vert_velocity, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);
        if (residual < m_pcg_residual_tolerance || iter >= m_pcg_max_iteration)
            break;
        
        Real z_dot_r = Calculate_z_dot_r();
        Real beta = (iter == 0) ? 0 : (z_dot_r / prev_z_dot_r);
        prev_z_dot_r = z_dot_r;
        SearchDirection(beta);
        NormalizeSearchDirection();
    
        cudaMemset(m_data.dev_vert_pAp, 0, sizeof(Real) * m_data.m_num_vert * 3);
        Calc_pAp_Elastic();
        Calc_pAp_GroundConstraint();
        Real pAp = Calculate_pAp();
        Real p_dot_r = Calculate_p_dot_r();
        alpha = p_dot_r / pAp;
    }

    PCG_After();
}

template <typename Real>
Real *PCGFEMSolver<Real>::GetDevicePositions()
{
    return this->m_data.dev_vert_position;
}

template <typename Real>
void PCGFEMSolver<Real>::PCG_Preparation()
{
    cudaMemcpy(m_data.dev_vert_position_prev, m_data.dev_vert_position, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);
    PCGFEMSolverKernel::initial_guess<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_data);
    cudaMemcpy(m_data.dev_vert_velocity_hat, m_data.dev_vert_p, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);
    cudaMemset(m_data.dev_vert_velocity_prev, 0, sizeof(Real) * m_data.m_num_vert * 3);
}

template <typename Real>
void PCGFEMSolver<Real>::UpdateSolution(Real alpha)
{
    thrust::transform(thrust::device_pointer_cast(m_data.dev_vert_velocity_prev),
                      thrust::device_pointer_cast(m_data.dev_vert_velocity_prev + m_data.m_num_vert * 3),
                      thrust::device_pointer_cast(m_data.dev_vert_p),
                      thrust::device_pointer_cast(m_data.dev_vert_velocity),
                      thrust::placeholders::_1 + alpha * thrust::placeholders::_2);
    thrust::transform(thrust::device_pointer_cast(m_data.dev_vert_position_prev),
                      thrust::device_pointer_cast(m_data.dev_vert_position_prev + m_data.m_num_vert * 3),
                      thrust::device_pointer_cast(m_data.dev_vert_velocity),
                      thrust::device_pointer_cast(m_data.dev_vert_position),
                      thrust::placeholders::_1 + m_data.m_time_step * thrust::placeholders::_2);
}

template <typename Real>
void PCGFEMSolver<Real>::ElasticForceAndPreconditioner()
{
    PCGFEMSolverKernel::tet_stiffness_matrix_diag<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_data);
    PCGFEMSolverKernel::calc_vert_diag_B<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_data);
    PCGFEMSolverKernel::calc_tetrahedron_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_data);
}

template <typename Real>
void PCGFEMSolver<Real>::GroundConstraintForceAndPreconditioner()
{
    cudaMemset(m_data.dev_vert_box_collision_A, 0, sizeof(Real) * m_data.m_num_vert * 3);
    PCGFEMSolverKernel::calc_ground_collision_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_data);
}

template <typename Real>
Real PCGFEMSolver<Real>::ResidualNorm()
{
    PCGFEMSolverKernel::vert_r_dot_r<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_data);
    Real residual = thrust::reduce(thrust::device_pointer_cast(m_data.dev_vert_temp3),
                                   thrust::device_pointer_cast(m_data.dev_vert_temp3 + m_data.m_num_vert * 3));
    residual = sqrt(residual / (m_data.m_num_vert * 3));
    return residual;
}

template <typename Real>
Real PCGFEMSolver<Real>::Calculate_z_dot_r()
{
    PCGFEMSolverKernel::vert_z_dot_r<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_data);
    Real z_dot_r = thrust::reduce(thrust::device_pointer_cast(m_data.dev_vert_temp3),
                                  thrust::device_pointer_cast(m_data.dev_vert_temp3 + m_data.m_num_vert * 3));
    return z_dot_r;
}

template <typename Real>
void PCGFEMSolver<Real>::SearchDirection(Real beta)
{
    PCGFEMSolverKernel::vert_search_direction<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_data, beta);
}

template <typename Real>
void PCGFEMSolver<Real>::NormalizeSearchDirection()
{
    thrust::transform(thrust::device_pointer_cast(m_data.dev_vert_p),
                      thrust::device_pointer_cast(m_data.dev_vert_p + m_data.m_num_vert * 3),
                      thrust::device_pointer_cast(m_data.dev_vert_temp3),
                      thrust::placeholders::_1 * thrust::placeholders::_1);
    Real p_dot_p = thrust::reduce(thrust::device_pointer_cast(m_data.dev_vert_temp3),
                                  thrust::device_pointer_cast(m_data.dev_vert_temp3 + m_data.m_num_vert * 3));
    thrust::transform(thrust::device_pointer_cast(m_data.dev_vert_p),
                      thrust::device_pointer_cast(m_data.dev_vert_p + m_data.m_num_vert * 3),
                      thrust::device_pointer_cast(m_data.dev_vert_temp3),
                      thrust::placeholders::_1 / sqrt(p_dot_p));
    cudaMemcpy(m_data.dev_vert_p, m_data.dev_vert_temp3, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);
}

template <typename Real>
void PCGFEMSolver<Real>::Calc_pAp_Elastic()
{
    cudaMemset(m_data.dev_vert_temp3, 0, sizeof(Real) * m_data.m_num_vert * 3);
    PCGFEMSolverKernel::calc_tet_Ap<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_data);
    PCGFEMSolverKernel::calc_elastic_pAp<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_data);
}

template <typename Real>
void PCGFEMSolver<Real>::Calc_pAp_GroundConstraint()
{
    PCGFEMSolverKernel::calc_ground_constraint_pAp<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert * 3), CUDA_BLOCK_SIZE>>>(m_data);
}

template <typename Real>
Real PCGFEMSolver<Real>::Calculate_pAp()
{
    return thrust::reduce(thrust::device_pointer_cast(m_data.dev_vert_pAp),
                          thrust::device_pointer_cast(m_data.dev_vert_pAp + m_data.m_num_vert * 3));
}

template <typename Real>
Real PCGFEMSolver<Real>::Calculate_p_dot_r()
{
    PCGFEMSolverKernel::vert_p_dot_r<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_data);
    return thrust::reduce(thrust::device_pointer_cast(m_data.dev_vert_temp3),
                          thrust::device_pointer_cast(m_data.dev_vert_temp3 + m_data.m_num_vert * 3));
}

template <typename Real>
void PCGFEMSolver<Real>::PCG_After()
{
}

template struct PCGFEMSolverData<float>;
template struct PCGFEMSolverData<double>;
template class PCGFEMSolver<float>;
template class PCGFEMSolver<double>;

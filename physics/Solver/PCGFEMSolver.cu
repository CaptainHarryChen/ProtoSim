#include "PCGFEMSolver.cuh"
#include <cuda_utils/cuda_utils.cuh>
#include <Math/algebra.cuh>
#include <Math/elastic_model.cuh>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

namespace PCGFEMSolverKernel
{
    template <typename Real>
    __global__ void tetrahedron_initialize(const PCGFEMSolverData<Real> *data)
    {
        unsigned int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= data->m_num_tet)
            return;
        unsigned int ind[4];
        for (unsigned int i = 0; i < 4; ++i)
            ind[i] = data->dev_tetrahedron[t * 4 + i] * 3;
        Real Dm[9];
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                Dm[i * 3 + j] = data->dev_vert_position[ind[j + 1] + i] - data->dev_vert_position[ind[0] + i];
        Real vol = cudaPhysics::det3(Dm);
        data->dev_tet_volume[t] = abs(vol) / 6.0f;
        cudaPhysics::matInv3(&data->dev_invDm[t * 9], Dm);
        for (unsigned int i = 0; i < 4; ++i)
            atomicAdd(&data->dev_vert_mass[data->dev_tetrahedron[t * 4 + i]], 0.25 * data->dev_tet_volume[t] * data->dev_tet_density[t]);
    }

    template <typename Real>
    __global__ void tet_stiffness_matrix_diag(const PCGFEMSolverData<Real> *data)
    {
        unsigned int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= data->m_num_tet)
            return;
        unsigned int ind[4];
        for (unsigned int i = 0; i < 4; ++i)
            ind[i] = data->dev_tetrahedron[t * 4 + i] * 3;
        Real K[12];
        cudaPhysics::calc_neohookean_force_diff<Real>(
            K,
            &data->dev_vert_position[ind[0]],
            &data->dev_vert_position[ind[1]],
            &data->dev_vert_position[ind[2]],
            &data->dev_vert_position[ind[3]],
            &data->dev_invDm[t * 9],
            data->dev_tet_volume[t],
            data->m_lame_mu,
            data->m_lame_lambda);
        for (unsigned int i = 0; i < 4; ++i)
        {
            for (unsigned int j = 0; j < 3; ++j)
            {
                // dev_vert_diag_B_const stores (K=partial f / partial x) instead of B for now
                atomicAdd(&data->dev_vert_diag_B_const[data->dev_tetrahedron[t * 4 + i] * 3 + j], K[i * 3 + j]);
            }
        }
    }

    template <typename Real>
    __global__ void calc_vert_diag_B_const(const PCGFEMSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        for (unsigned int i = 0; i < 3; ++i)
        {
            data->dev_vert_diag_B_const[3 * v + i] = data->dev_vert_mass[v] * data->m_time_step_inv - data->dev_vert_diag_B_const[3 * v + i] * data->m_time_step;
        }
    }

    template <typename Real>
    __global__ void initial_guess(const PCGFEMSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        cudaPhysics::axpby(&data->dev_vert_velocity[3 * v], (Real)1.0, &data->dev_vert_velocity[3 * v], data->m_time_step, data->dev_gravity, 3);
        cudaPhysics::axpby(&data->dev_vert_position_next[3 * v], (Real)1.0, &data->dev_vert_position[3 * v], data->m_time_step, &data->dev_vert_velocity[3 * v], 3);
    }

    template <typename Real>
    __global__ void calc_tetrahedron_force(const PCGFEMSolverData<Real> *data)
    {
        unsigned int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= data->m_num_tet)
            return;
        unsigned int ind[4];
        for (unsigned int i = 0; i < 4; ++i)
            ind[i] = data->dev_tetrahedron[t * 4 + i] * 3;
        
        Real Ds[9];
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                Ds[i * 3 + j] = data->dev_vert_position_next[ind[j + 1] + i] - data->dev_vert_position_next[ind[0] + i];
        const Real *InvDm = &data->dev_invDm[t * 9];
        Real F[9], P[9], H[9], f0[3] = {0};
        cudaPhysics::matMul3(F, Ds, InvDm);
        cudaPhysics::calc_neohookean_P<Real>(P, F, data->m_lame_mu, data->m_lame_lambda);
        cudaPhysics::matmatTMul3(H, P, InvDm);
        cudaPhysics::vecMul(H, -data->dev_tet_volume[t], H, 9);
        for (unsigned int i = 0; i < 3; ++i)
            cudaPhysics::vecSubs3(f0, f0, &H[i * 3]);
        
        for (unsigned int j = 0; j < 3; ++j)
            atomicAdd(&data->dev_vert_force[ind[0] + j], f0[j]);
        for (unsigned int i = 1; i < 4; ++i)
        {
            for (unsigned int j = 0; j < 3; ++j)
                atomicAdd(&data->dev_vert_force[ind[i] + j], H[(i - 1) * 3 + j]);
        }
    }

    template <typename Real>
    __global__ void calc_ground_collision_force(const PCGFEMSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        if (data->dev_vert_position_next[3 * v + 1] < 0.0f)
        {
            data->dev_vert_force[3 * v + 1] += -data->m_ground_collision_stiffness * data->dev_vert_position_next[3 * v + 1] * data->dev_vert_mass[v];
            data->dev_vert_diag_B_mutable[3 * v + 1] += data->m_ground_collision_stiffness * data->dev_vert_mass[v] * data->m_time_step;
        }
    }

    template <typename Real>
    __global__ void vert_r_dot_r(const PCGFEMSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            Real r = data->dev_vert_mass[v] * data->m_time_step_inv * (data->dev_vert_velocity[3 * v + j] - data->dev_vert_velocity_hat[3 * v + j]) - data->dev_vert_force[3 * v + j];
            data->dev_vert_temp[3 * v + j] = r * r;
        }
    }

    template <typename Real>
    __global__ void vert_z_dot_r(const PCGFEMSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            Real r = data->dev_vert_mass[v] * data->m_time_step_inv * (data->dev_vert_velocity[3 * v + j] - data->dev_vert_velocity_hat[3 * v + j]) - data->dev_vert_force[3 * v + j];
            data->dev_vert_temp[3 * v + j] = r * r / (data->dev_vert_diag_B_const[3 * v + j] + data->dev_vert_diag_B_mutable[3 * v + j]);
        }
    }

    template <typename Real>
    __global__ void vert_search_direction(const PCGFEMSolverData<Real> *data, Real beta)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            Real r = data->dev_vert_mass[v] * data->m_time_step_inv * (data->dev_vert_velocity[3 * v + j] - data->dev_vert_velocity_hat[3 * v + j]) - data->dev_vert_force[3 * v + j];
            data->dev_vert_p[v * 3 + j] = - r / (data->dev_vert_diag_B_const[v * 3 + j] + data->dev_vert_diag_B_mutable[v * 3 + j])
                                          + beta * data->dev_vert_p[v * 3 + j];
        }
    }

    template <typename Real>
    __global__ void calc_tet_Ap(const PCGFEMSolverData<Real> *data)
    {
        unsigned int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= data->m_num_tet)
            return;
        unsigned int ind[4];
        for (unsigned int i = 0; i < 4; ++i)
            ind[i] = data->dev_tetrahedron[t * 4 + i] * 3;

        Real Ds[9], partial_Ds[9];
        const Real *InvDm = &data->dev_invDm[t * 9];
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                Ds[i * 3 + j] = data->dev_vert_position_next[ind[j + 1] + i] - data->dev_vert_position_next[ind[0] + i];
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                partial_Ds[i * 3 + j] = (data->dev_vert_p[ind[j + 1] + i] - data->dev_vert_p[ind[0] + i]) * data->m_time_step;
        
        Real F[9], partial_F[9], F_inv[9], J;
        cudaPhysics::matMul3(F, Ds, InvDm);
        cudaPhysics::matMul3(partial_F, partial_Ds, InvDm);
        cudaPhysics::matInv3(F_inv, F);
        J = cudaPhysics::det3(F);

        Real P1[9], P2[9], P2_T[9], F_inv_mul_partial_F[9], P3[9], P3_T[9];
        cudaPhysics::vecMul(P1, data->m_lame_mu, partial_F, 9);

        cudaPhysics::matMul3(F_inv_mul_partial_F, F_inv, partial_F);
        cudaPhysics::matmatTMul3(P2_T, F_inv_mul_partial_F, F_inv);
        cudaPhysics::matTrans3(P2, P2_T);
        cudaPhysics::vecMul(P2, data->m_lame_mu - data->m_lame_lambda * log(J), P2, 9);

        Real trace = F_inv_mul_partial_F[0] + F_inv_mul_partial_F[4] + F_inv_mul_partial_F[8];
        cudaPhysics::vecMul(P3_T, data->m_lame_lambda * trace, F_inv, 9);
        cudaPhysics::matTrans3(P3, P3_T);

        Real partial_P[9], partial_H[9];
        cudaPhysics::axpbypcz(partial_P, (Real)1.0, P1, (Real)1.0, P2, (Real)1.0, P3, 9);
        cudaPhysics::matmatTMul3(partial_H, partial_P, InvDm);
        cudaPhysics::vecMul(partial_H, -data->dev_tet_volume[t], partial_H, 9);

        Real partial_f0[3] = {0};
        for (unsigned int i = 0; i < 3; ++i)
            cudaPhysics::vecSubs3(partial_f0, partial_f0, &partial_H[i * 3]);

        for (unsigned int j = 0; j < 3; ++j)
            atomicAdd(&data->dev_vert_temp[ind[0] + j], partial_f0[j]);
        for (unsigned int i = 1; i < 4; ++i)
        {
            for (unsigned int j = 0; j < 3; ++j)
                atomicAdd(&data->dev_vert_temp[ind[i] + j], partial_H[(i - 1) * 3 + j]);
        }
    }
    
    template <typename Real>
    __global__ void calc_vert_pAp(const PCGFEMSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            Real Ap =   data->dev_vert_mass[v] * data->m_time_step_inv * data->dev_vert_p[3 * v + j] 
                      - data->dev_vert_temp[3 * v + j] // elastic force part stored in dev_vert_temp
                      + data->dev_vert_diag_B_mutable[3 * v + j] * data->dev_vert_p[3 * v + j]; // ground collision part
            data->dev_vert_temp[3 * v + j] = data->dev_vert_p[3 * v + j] * Ap;
        }
    }

    template <typename Real>
    __global__ void vert_p_dot_r(const PCGFEMSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            Real r = data->dev_vert_mass[v] * data->m_time_step_inv * (data->dev_vert_velocity[3 * v + j] - data->dev_vert_velocity_hat[3 * v + j]) - data->dev_vert_force[3 * v + j];
            data->dev_vert_temp[3 * v + j] = data->dev_vert_p[3 * v + j] * r;
        }
    }
}

template <typename Real>
PCGFEMSolver<Real>::PCGFEMSolver(const std::vector<Real> &position, const std::vector<unsigned int> &tetrahedron, const std::vector<Real> &tetrahedron_density)
{
    m_data.m_num_vert = (unsigned int)position.size() / 3;
    m_data.m_num_tet = (unsigned int)tetrahedron.size() / 4;

    printf("PCGFEMSolver: num_vert=%u, num_tet=%u\n", m_data.m_num_vert, m_data.m_num_tet);

    cudaMalloc((void **)&m_data.dev_vert_position, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemcpy(m_data.dev_vert_position, position.data(), sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_vert_position_next, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_vert_velocity, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemset(m_data.dev_vert_velocity, 0, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_vert_velocity_hat, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_vert_mass, sizeof(Real) * m_data.m_num_vert);
    cudaMemset(m_data.dev_vert_mass, 0, sizeof(Real) * m_data.m_num_vert);
    cudaMalloc((void **)&m_data.dev_vert_force, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_vert_diag_B_const, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemset(m_data.dev_vert_diag_B_const, 0, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_vert_diag_B_mutable, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_vert_p, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_vert_temp, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemset(m_data.dev_vert_temp, 0, sizeof(Real) * m_data.m_num_vert * 3);

    cudaMalloc((void **)&m_data.dev_tetrahedron, sizeof(unsigned int) * tetrahedron.size());
    cudaMemcpy(m_data.dev_tetrahedron, tetrahedron.data(), sizeof(unsigned int) * tetrahedron.size(), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_tet_density, sizeof(Real) * m_data.m_num_tet);
    cudaMemcpy(m_data.dev_tet_density, tetrahedron_density.data(), sizeof(Real) * m_data.m_num_tet, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_tet_volume, sizeof(Real) * m_data.m_num_tet);
    cudaMemset(m_data.dev_tet_volume, 0, sizeof(Real) * m_data.m_num_tet);
    cudaMalloc((void **)&m_data.dev_invDm, sizeof(Real) * m_data.m_num_tet * 9);

    m_data.m_time_step = TIME_STEP;
    m_data.m_time_step_inv = 1.0f / m_data.m_time_step;
    m_data.m_lame_mu = LAME_MU;
    m_data.m_lame_lambda = LAME_LAMBDA;
    m_data.m_ground_collision_stiffness = GROUND_COLLISION_STIFFNESS;
    cudaMalloc(&m_data.dev_gravity, sizeof(Real) * 3);
    std::vector<Real> gravity = {0.0f, -GRAVITY, 0.0f};
    cudaMemcpy(m_data.dev_gravity, gravity.data(), sizeof(Real) * 3, cudaMemcpyHostToDevice);

    cudaMalloc(&m_dev_data, sizeof(PCGFEMSolverData<Real>));
    cudaMemcpy(m_dev_data, &m_data, sizeof(PCGFEMSolverData<Real>), cudaMemcpyHostToDevice);

    PCGFEMSolverKernel::tetrahedron_initialize<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
    PCGFEMSolverKernel::tet_stiffness_matrix_diag<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
    PCGFEMSolverKernel::calc_vert_diag_B_const<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaCheck(cudaDeviceSynchronize());
    printf("PCGFEMSolver initialized.\n");
}

template <typename Real>
PCGFEMSolver<Real>::~PCGFEMSolver()
{
    cudaFree(m_data.dev_vert_position);
    cudaFree(m_data.dev_vert_position_next);
    cudaFree(m_data.dev_vert_velocity);
    cudaFree(m_data.dev_vert_velocity_hat);
    cudaFree(m_data.dev_vert_mass);
    cudaFree(m_data.dev_vert_force);
    cudaFree(m_data.dev_vert_diag_B_const);
    cudaFree(m_data.dev_vert_diag_B_mutable);
    cudaFree(m_data.dev_vert_p);
    cudaFree(m_data.dev_vert_temp);

    cudaFree(m_data.dev_tetrahedron);
    cudaFree(m_data.dev_tet_density);
    cudaFree(m_data.dev_tet_volume);
    cudaFree(m_data.dev_invDm);

    cudaFree(m_data.dev_gravity);

    cudaFree(m_dev_data);
}

template <typename Real>
void PCGFEMSolver<Real>::Step()
{
    PCGFEMSolverKernel::initial_guess<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaMemcpy(m_data.dev_vert_velocity_hat, m_data.dev_vert_velocity, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);

    Real prev_z_dot_r = 0; // for calculate beta in PCG
    for (unsigned int iter = 0; iter < MAX_ITERATIONS; ++iter)
    {
        if (m_verbose)
            printf("PCG iteration %u\n", iter);

        cudaMemset(m_data.dev_vert_force, 0, sizeof(Real) * m_data.m_num_vert * 3);
        cudaMemset(m_data.dev_vert_diag_B_mutable, 0, sizeof(Real) * m_data.m_num_vert * 3);
        PCGFEMSolverKernel::calc_tetrahedron_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
        PCGFEMSolverKernel::calc_ground_collision_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);

        PCGFEMSolverKernel::vert_r_dot_r<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
        Real residual = thrust::reduce(thrust::device_pointer_cast(m_data.dev_vert_temp),
                                       thrust::device_pointer_cast(m_data.dev_vert_temp + m_data.m_num_vert * 3)) / m_data.m_num_vert;
        if (m_verbose)
            printf("  residual = %e\n", residual);
        if (residual < RESIDUAL_TOLERANCE)
            break;
        
        Real beta = 0;
        PCGFEMSolverKernel::vert_z_dot_r<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
        Real z_dot_r = thrust::reduce(thrust::device_pointer_cast(m_data.dev_vert_temp),
                                      thrust::device_pointer_cast(m_data.dev_vert_temp + m_data.m_num_vert * 3));
        if (iter > 0)
            beta = z_dot_r / prev_z_dot_r;
        prev_z_dot_r = z_dot_r;
        PCGFEMSolverKernel::vert_search_direction<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data, beta);

        thrust::transform(thrust::device_pointer_cast(m_data.dev_vert_p),
                          thrust::device_pointer_cast(m_data.dev_vert_p + m_data.m_num_vert * 3),
                          thrust::device_pointer_cast(m_data.dev_vert_temp),
                          thrust::placeholders::_1 * thrust::placeholders::_1);
        Real p_dot_p = thrust::reduce(thrust::device_pointer_cast(m_data.dev_vert_temp),
                                      thrust::device_pointer_cast(m_data.dev_vert_temp + m_data.m_num_vert * 3));
        thrust::transform(thrust::device_pointer_cast(m_data.dev_vert_p),
                          thrust::device_pointer_cast(m_data.dev_vert_p + m_data.m_num_vert * 3),
                          thrust::device_pointer_cast(m_data.dev_vert_temp),
                          thrust::placeholders::_1 / sqrt(p_dot_p));
        cudaMemcpy(m_data.dev_vert_p, m_data.dev_vert_temp, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);
    
        cudaMemset(m_data.dev_vert_temp, 0, sizeof(Real) * m_data.m_num_vert * 3);
        PCGFEMSolverKernel::calc_tet_Ap<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
        PCGFEMSolverKernel::calc_vert_pAp<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
        Real pAp = thrust::reduce(thrust::device_pointer_cast(m_data.dev_vert_temp),
                                  thrust::device_pointer_cast(m_data.dev_vert_temp + m_data.m_num_vert * 3));
        PCGFEMSolverKernel::vert_p_dot_r<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
        Real alpha = - thrust::reduce(thrust::device_pointer_cast(m_data.dev_vert_temp),
                                      thrust::device_pointer_cast(m_data.dev_vert_temp + m_data.m_num_vert * 3)) / pAp;

        thrust::transform(thrust::device_pointer_cast(m_data.dev_vert_velocity),
                          thrust::device_pointer_cast(m_data.dev_vert_velocity + m_data.m_num_vert * 3),
                          thrust::device_pointer_cast(m_data.dev_vert_p),
                          thrust::device_pointer_cast(m_data.dev_vert_velocity),
                          thrust::placeholders::_1 + alpha * thrust::placeholders::_2);
        thrust::transform(thrust::device_pointer_cast(m_data.dev_vert_position_next),
                          thrust::device_pointer_cast(m_data.dev_vert_position_next + m_data.m_num_vert * 3),
                          thrust::device_pointer_cast(m_data.dev_vert_velocity),
                          thrust::device_pointer_cast(m_data.dev_vert_position_next),
                          thrust::placeholders::_1 + m_data.m_time_step * thrust::placeholders::_2);
    }

    cudaMemcpy(m_data.dev_vert_position, m_data.dev_vert_position_next, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);
}

template <typename Real>
Real *PCGFEMSolver<Real>::GetDevicePositions()
{
    return this->m_data.dev_vert_position;
}

template struct PCGFEMSolverData<float>;
template struct PCGFEMSolverData<double>;
template class PCGFEMSolver<float>;
template class PCGFEMSolver<double>;

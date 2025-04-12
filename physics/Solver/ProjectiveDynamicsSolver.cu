#include "ProjectiveDynamicsSolver.cuh"
#include <cuda_utils/cuda_utils.cuh>
#include <Math/algebra.cuh>
#include <Math/elastic_model.cuh>

namespace ProjectiveDynamicsSolverKernel
{
    template <typename Real>
    __global__ void tetrahedron_initialize(ProjectiveDynamicsSolverData<Real> *data)
    {
        unsigned int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= data->m_num_tet)
            return;
        unsigned int ind[4];
        for (unsigned int i = 0; i < 4; ++i)
            ind[i] = data->dev_tetrahedron[t * 4 + i] * 3;
        Real Dm[9];
        Dm[0] = data->dev_position[ind[1] + 0] - data->dev_position[ind[0] + 0];
        Dm[3] = data->dev_position[ind[1] + 1] - data->dev_position[ind[0] + 1];
        Dm[6] = data->dev_position[ind[1] + 2] - data->dev_position[ind[0] + 2];
        Dm[1] = data->dev_position[ind[2] + 0] - data->dev_position[ind[0] + 0];
        Dm[4] = data->dev_position[ind[2] + 1] - data->dev_position[ind[0] + 1];
        Dm[7] = data->dev_position[ind[2] + 2] - data->dev_position[ind[0] + 2];
        Dm[2] = data->dev_position[ind[3] + 0] - data->dev_position[ind[0] + 0];
        Dm[5] = data->dev_position[ind[3] + 1] - data->dev_position[ind[0] + 1];
        Dm[8] = data->dev_position[ind[3] + 2] - data->dev_position[ind[0] + 2];
        Real vol = cudaPhysics::det3(Dm);
        data->dev_tet_volume[t] = abs(vol) / 6.0f;
        cudaPhysics::matInv3(&data->dev_invDm[t * 9], Dm);
        for (unsigned int i = 0; i < 4; ++i)
            atomicAdd(&data->dev_mass[data->dev_tetrahedron[t * 4 + i]], 0.25 * data->dev_tet_volume[t] * data->dev_tet_density[t]);
    }

    template <typename Real>
    __global__ void jacobi_precondition(ProjectiveDynamicsSolverData<Real> *data)
    {
        unsigned int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= data->m_num_tet)
            return;
        unsigned int *v = &data->dev_tetrahedron[4 * t];
        Real *invDm = &data->dev_invDm[9 * t];
        Real invDmT[9];
        cudaPhysics::matTrans3(invDmT, invDm);
        Real delta_f[9]; // delta_f = -2 * V0 * stiffness * invDm * invDm^T * (delta Ds^T)
        cudaPhysics::matMul3(delta_f, invDm, invDmT);
        cudaPhysics::vecMul(delta_f, -2 * data->dev_tet_volume[t] * data->m_stiffness, delta_f, 9);
        Real H[4];
        H[1] = delta_f[0]; //(delta Ds^T)[0][0:3] = 1
        H[2] = delta_f[4]; //(delta Ds^T)[1][0:3] = 1
        H[3] = delta_f[8]; //(delta Ds^T)[2][0:3] = 1
        // (delta Ds^T)[:,:] = -1 and this need a sum
        H[0] = delta_f[0] + delta_f[1] + delta_f[2] + delta_f[3] + delta_f[4] + delta_f[5] + delta_f[6] + delta_f[7] + delta_f[8];
        for (unsigned int i = 0; i < 4; ++i)
            atomicAdd(&data->dev_diag_Hessian[v[i]], -H[i]);
    }

    template <typename Real>
    __global__ void initial_guess(ProjectiveDynamicsSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        cudaPhysics::accumulate(&data->dev_velocity[3 * v], data->m_time_step, data->dev_gravity, 3);
        cudaPhysics::accumulate(&data->dev_position[3 * v], data->m_time_step, &data->dev_velocity[3 * v], 3);
    }

    // compute h^{-2} * mass only once outside the loop
    template <typename Real>
    __global__ void compute_init_AB(ProjectiveDynamicsSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        data->dev_init_A[v] = data->dev_mass[v] * data->m_time_step_inv * data->m_time_step_inv;
        cudaPhysics::vecMul3(&data->dev_init_B[3 * v], data->dev_init_A[v], &data->dev_position[3 * v]);
    }

    template <typename Real>
    __global__ void ground_collision_detection(ProjectiveDynamicsSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        if (data->dev_position[3 * v + 1] < 0.0f)
        {
            unsigned int k = atomicAdd(data->dev_ground_collision_count, 1);
            data->dev_ground_collision_ids[k] = v;
        }
    }

    template <typename Real>
    __global__ void compute_ground_collision_force(ProjectiveDynamicsSolverData<Real> *data)
    {
        unsigned int c = blockDim.x * blockIdx.x + threadIdx.x;
        if (c >= *data->dev_ground_collision_count)
            return;
        unsigned int v = data->dev_ground_collision_ids[c];
        if (data->dev_position[3 * v + 1] < 0.0f)
        {
            data->dev_vert_force[3 * v + 1] += -data->m_stiffness * data->dev_position[3 * v + 1];
            data->dev_vert_Hessian[v] += data->m_stiffness;
        }
    }

    template <typename Real>
    __global__ void compute_iteration(ProjectiveDynamicsSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        Real b[3];
        cudaPhysics::axpbypcz(b, 3, (Real)1.0, &data->dev_init_B[3 * v], -data->dev_init_A[v], &data->dev_position[3 * v], (Real)1.0, &data->dev_vert_force[3 * v]);
        for (unsigned int i = data->dev_vert_to_tet_offset[v]; i < data->dev_vert_to_tet_offset[v + 1]; ++i)
        {
            unsigned int offset;
            for (unsigned int j = 0; j < 4; ++j)
            {
                if (v == data->dev_tetrahedron[4 * data->dev_vert_to_tet[i] + j])
                {
                    offset = j;
                    break;
                }
            }
            cudaPhysics::accumulate(b, (Real)1.0, &data->dev_tet_force[12 * data->dev_vert_to_tet[i] + 3 * offset], 3);
        }
        cudaPhysics::vecMul3(&data->dev_position_delta[3 * v], (Real)1.0 / (data->dev_init_A[v] + data->dev_diag_Hessian[v] + data->dev_vert_Hessian[v]), b);
    }

    template <typename Real>
    __global__ void Chebyshev(ProjectiveDynamicsSolverData<Real> *data, Real omega)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        Real delta_position[3];
        cudaPhysics::vecSubs3(delta_position, &data->dev_position_next[3 * v], &data->dev_position[3 * v]);
        cudaPhysics::axpby(&data->dev_position_next[3 * v], 3, data->m_under_relaxation, delta_position, (Real)1.0, &data->dev_position[3 * v]);
        cudaPhysics::vecSubs3(delta_position, &data->dev_position_next[3 * v], &data->dev_position_prev[3 * v]);
        cudaPhysics::axpby(&data->dev_position_next[3 * v], 3, omega, delta_position, (Real)1.0, &data->dev_position_prev[3 * v]);
    }

    template <typename Real>
    __global__ void update_velocity(ProjectiveDynamicsSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        cudaPhysics::vecSubs3(&data->dev_velocity[3 * v], &data->dev_position[3 * v], &data->dev_position_backup[3 * v]);
        cudaPhysics::vecMul3(&data->dev_velocity[3 * v], data->m_time_step_inv, &data->dev_velocity[3 * v]);
    }

    template <typename Real>
    __global__ void projective_dynamics_constraint(ProjectiveDynamicsSolverData<Real> *data)
    {
        unsigned int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= data->m_num_tet)
            return;
        int ind[4];
        for (unsigned int i = 0; i < 4; ++i)
        {
            ind[i] = data->dev_tetrahedron[t * 4 + i] * 3;
        }

        Real Ds[9];
        Real *idm = &data->dev_invDm[t * 9];
        Ds[0] = data->dev_position[ind[1] + 0] - data->dev_position[ind[0] + 0];
        Ds[3] = data->dev_position[ind[1] + 1] - data->dev_position[ind[0] + 1];
        Ds[6] = data->dev_position[ind[1] + 2] - data->dev_position[ind[0] + 2];
        Ds[1] = data->dev_position[ind[2] + 0] - data->dev_position[ind[0] + 0];
        Ds[4] = data->dev_position[ind[2] + 1] - data->dev_position[ind[0] + 1];
        Ds[7] = data->dev_position[ind[2] + 2] - data->dev_position[ind[0] + 2];
        Ds[2] = data->dev_position[ind[3] + 0] - data->dev_position[ind[0] + 0];
        Ds[5] = data->dev_position[ind[3] + 1] - data->dev_position[ind[0] + 1];
        Ds[8] = data->dev_position[ind[3] + 2] - data->dev_position[ind[0] + 2];
        Real F[9], R[9];
        cudaPhysics::matMul3(F, Ds, idm);
        cudaPhysics::get_rotation_matrix_from_deformation_gradient(R, F);

        Real f[9]; // f = -2 * V0 * stiffness * invDm * (F - R)^T
        Real FSubR[9], FSubRT[9];
        cudaPhysics::vecSubs(FSubR, F, R, 9);
        cudaPhysics::matTrans3(FSubRT, FSubR);
        cudaPhysics::matMul3(f, idm, FSubRT);
        cudaPhysics::vecMul(f, -2 * data->dev_tet_volume[t] * data->m_stiffness, f, 9);
        cudaPhysics::vecCopy(&data->dev_tet_force[12 * t + 3], f, 9);
        cudaPhysics::axpbypcz(&data->dev_tet_force[12 * t], 3, (Real)-1.0, &f[0], (Real)-1.0, &f[3], (Real)-1.0, &f[6]);
    }

    template <typename Real>
    __global__ void update_position(ProjectiveDynamicsSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        cudaPhysics::vecAdd3(&data->dev_position_next[3 * v], &data->dev_position[3 * v], &data->dev_position_delta[3 * v]);
    }

    template <typename Real>
    __global__ void swap_position(ProjectiveDynamicsSolverData<Real> *data)
    {
        Real *temp = data->dev_position;
        data->dev_position = data->dev_position_prev;
        data->dev_position_prev = temp;
        temp = data->dev_position;
        data->dev_position = data->dev_position_next;
        data->dev_position_next = temp;
    }
}

template <typename Real>
ProjectiveDynamicsSolver<Real>::ProjectiveDynamicsSolver(const std::vector<Real> &position, const std::vector<unsigned int> &tetrahedron, const std::vector<Real> &tetrahedron_density,
                                                         const std::vector<unsigned int> &object_tetrahedron_offset)
{
    m_data.m_num_vert = (unsigned int)position.size() / 3;
    m_data.m_num_tet = (unsigned int)tetrahedron.size() / 4;

    cudaMalloc((void **)&m_data.dev_position, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemcpy(m_data.dev_position, position.data(), sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_position_prev, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemcpy(m_data.dev_position_prev, position.data(), sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_position_next, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_position_backup, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_position_delta, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_velocity, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemset(m_data.dev_velocity, 0, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_mass, sizeof(Real) * m_data.m_num_vert);
    cudaMemset(m_data.dev_mass, 0, sizeof(Real) * m_data.m_num_vert);
    cudaMalloc((void **)&m_data.dev_mass_inv, sizeof(Real) * m_data.m_num_vert);
    cudaMalloc((void **)&m_data.dev_inertia, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_init_A, sizeof(Real) * m_data.m_num_vert);
    cudaMalloc((void **)&m_data.dev_init_B, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_vert_force, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_vert_Hessian, sizeof(Real) * m_data.m_num_vert);
    cudaMalloc((void **)&m_data.dev_diag_Hessian, sizeof(Real) * m_data.m_num_vert);
    cudaMemset(m_data.dev_diag_Hessian, 0, sizeof(Real) * m_data.m_num_vert);

    std::vector<unsigned int> vert_to_tet;
    std::vector<unsigned int> vert_to_tet_offset;
    std::vector<std::vector<unsigned int>> vert_to_tet_temp(m_data.m_num_vert);
    for (unsigned int i = 0; i < m_data.m_num_tet; ++i)
        for (unsigned int j = 0; j < 4; ++j)
            vert_to_tet_temp[tetrahedron[i * 4 + j]].push_back(i);
    vert_to_tet_offset.push_back(0);
    for (unsigned int i = 0; i < m_data.m_num_vert; ++i)
    {
        vert_to_tet_offset.push_back(vert_to_tet_offset.back() + (unsigned int)vert_to_tet_temp[i].size());
        vert_to_tet.insert(vert_to_tet.end(), vert_to_tet_temp[i].begin(), vert_to_tet_temp[i].end());
    }
    cudaMalloc((void **)&m_data.dev_vert_to_tet, sizeof(unsigned int) * vert_to_tet.size());
    cudaMemcpy(m_data.dev_vert_to_tet, vert_to_tet.data(), sizeof(unsigned int) * vert_to_tet.size(), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_vert_to_tet_offset, sizeof(unsigned int) * vert_to_tet_offset.size());
    cudaMemcpy(m_data.dev_vert_to_tet_offset, vert_to_tet_offset.data(), sizeof(unsigned int) * vert_to_tet_offset.size(), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&m_data.dev_tetrahedron, sizeof(unsigned int) * tetrahedron.size());
    cudaMemcpy(m_data.dev_tetrahedron, tetrahedron.data(), sizeof(unsigned int) * tetrahedron.size(), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_tet_density, sizeof(Real) * m_data.m_num_tet);
    cudaMemcpy(m_data.dev_tet_density, tetrahedron_density.data(), sizeof(Real) * m_data.m_num_tet, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_tet_volume, sizeof(Real) * m_data.m_num_tet);
    cudaMemset(m_data.dev_tet_volume, 0, sizeof(Real) * m_data.m_num_tet);
    cudaMalloc((void **)&m_data.dev_tet_force, sizeof(Real) * m_data.m_num_tet * 12);
    cudaMemset(m_data.dev_tet_force, 0, sizeof(Real) * m_data.m_num_tet * 12);
    cudaMalloc((void **)&m_data.dev_invDm, sizeof(Real) * m_data.m_num_tet * 9);

    cudaMalloc((void **)&m_data.dev_ground_collision_count, sizeof(unsigned int));
    cudaMalloc((void **)&m_data.dev_ground_collision_ids, sizeof(unsigned int) * m_data.m_num_vert);

    m_data.m_time_step = TIME_STEP;
    m_data.m_time_step_inv = 1.0f / m_data.m_time_step;
    m_data.m_stiffness = LAME_MU;
    m_data.m_under_relaxation = UNDER_RELAXATION;
    cudaMalloc(&m_data.dev_gravity, sizeof(Real) * 3);
    std::vector<Real> gravity = {0.0f, -GRAVITY, 0.0f};
    cudaMemcpy(m_data.dev_gravity, gravity.data(), sizeof(Real) * 3, cudaMemcpyHostToDevice);

    cudaMalloc(&m_dev_data, sizeof(ProjectiveDynamicsSolverData<Real>));
    cudaMemcpy(m_dev_data, &m_data, sizeof(ProjectiveDynamicsSolverData<Real>), cudaMemcpyHostToDevice);

    ProjectiveDynamicsSolverKernel::tetrahedron_initialize<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaPhysics::array_real_inv<Real>(m_data.dev_mass_inv, m_data.dev_mass, m_data.m_num_vert);
    ProjectiveDynamicsSolverKernel::jacobi_precondition<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
}

template <typename Real>
ProjectiveDynamicsSolver<Real>::~ProjectiveDynamicsSolver()
{
    cudaFree(m_data.dev_position);
    cudaFree(m_data.dev_position_prev);
    cudaFree(m_data.dev_position_next);
    cudaFree(m_data.dev_position_backup);
    cudaFree(m_data.dev_position_delta);
    cudaFree(m_data.dev_velocity);
    cudaFree(m_data.dev_mass);
    cudaFree(m_data.dev_mass_inv);
    cudaFree(m_data.dev_inertia);
    cudaFree(m_data.dev_init_A);
    cudaFree(m_data.dev_init_B);
    cudaFree(m_data.dev_vert_force);
    cudaFree(m_data.dev_vert_Hessian);
    cudaFree(m_data.dev_diag_Hessian);

    cudaFree(m_data.dev_vert_to_tet);
    cudaFree(m_data.dev_vert_to_tet_offset);

    cudaFree(m_data.dev_tetrahedron);
    cudaFree(m_data.dev_tet_density);
    cudaFree(m_data.dev_tet_volume);
    cudaFree(m_data.dev_tet_force);
    cudaFree(m_data.dev_invDm);

    cudaFree(m_data.dev_ground_collision_count);
    cudaFree(m_data.dev_ground_collision_ids);

    cudaFree(m_data.dev_gravity);

    cudaFree(m_dev_data);
}

template <typename Real>
void ProjectiveDynamicsSolver<Real>::Step()
{
    cudaMemcpy(m_data.dev_position_backup, m_data.dev_position, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);
    ProjectiveDynamicsSolverKernel::initial_guess<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaMemcpy(m_data.dev_inertia, m_data.dev_position, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);
    ProjectiveDynamicsSolverKernel::compute_init_AB<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);

    Real omega = 1;
    for (unsigned int iter = 0; iter < MAX_ITERATIONS; ++iter)
    {
        ProjectiveDynamicsSolverKernel::projective_dynamics_constraint<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
        cudaMemset(m_data.dev_ground_collision_count, 0, sizeof(unsigned int));
        ProjectiveDynamicsSolverKernel::ground_collision_detection<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
        unsigned int ground_collision_count;
        cudaMemcpy(&ground_collision_count, m_data.dev_ground_collision_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemset(m_data.dev_vert_force, 0, sizeof(Real) * m_data.m_num_vert * 3);
        cudaMemset(m_data.dev_vert_Hessian, 0, sizeof(Real) * m_data.m_num_vert);
        ProjectiveDynamicsSolverKernel::compute_ground_collision_force<Real><<<CUDA_GRID_SIZE(ground_collision_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
        ProjectiveDynamicsSolverKernel::compute_iteration<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
        ProjectiveDynamicsSolverKernel::update_position<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
        UpdateChebyshevOmega(omega, iter);
        ProjectiveDynamicsSolverKernel::Chebyshev<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data, omega);
        std::swap(m_data.dev_position, m_data.dev_position_prev);
        std::swap(m_data.dev_position, m_data.dev_position_next);
        ProjectiveDynamicsSolverKernel::swap_position<Real><<<1, 1>>>(m_dev_data);
    }
    ProjectiveDynamicsSolverKernel::update_velocity<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
}

template <typename Real>
Real *ProjectiveDynamicsSolver<Real>::GetDevicePositions()
{
    return this->m_data.dev_position;
}

template <typename Real>
void ProjectiveDynamicsSolver<Real>::UpdateChebyshevOmega(Real &omega, unsigned int iter)
{
    if (iter < CHEBYSHEV_DELAY_ITER)
        omega = 1;
    else if (iter == CHEBYSHEV_DELAY_ITER)
        omega = 2 / (2 - CHEBYSHEV_RHO * CHEBYSHEV_RHO);
    else
        omega = 4 / (4 - CHEBYSHEV_RHO * CHEBYSHEV_RHO * omega);
}

template struct ProjectiveDynamicsSolverData<float>;
template struct ProjectiveDynamicsSolverData<double>;
template class ProjectiveDynamicsSolver<float>;
template class ProjectiveDynamicsSolver<double>;

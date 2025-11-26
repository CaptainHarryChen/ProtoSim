#include "ProjectiveDynamicsSolver.cuh"
#include <cuda_utils/cuda_utils.cuh>
#include <Math/algebra.cuh>
#include <Math/ConstitutiveModel/CorotatedLinear.cuh>

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
    __global__ void tet_stiffness_matrix_diag(ProjectiveDynamicsSolverData<Real> *data)
    {
        unsigned int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= data->m_num_tet)
            return;
        unsigned int *v = &data->dev_tetrahedron[4 * t];
        Real *invDm = &data->dev_invDm[9 * t];
        Real K_diag[4];
        cudaPhysics::calc_corotated_linear_K_diag<Real>(K_diag, invDm, data->dev_tet_volume[t], data->m_lame_mu, data->m_lame_lambda);
        for (unsigned int i = 0; i < 4; ++i)
            atomicAdd(&data->dev_stiffness_matrix_diag[v[i]], K_diag[i]);
    }

    template <typename Real>
    __global__ void initial_guess(ProjectiveDynamicsSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        cudaPhysics::axpby(&data->dev_velocity[3 * v], (Real)1.0, &data->dev_velocity[3 * v], data->m_time_step, data->dev_gravity, 3);
        cudaPhysics::axpby(&data->dev_position[3 * v], (Real)1.0, &data->dev_position[3 * v], data->m_time_step, &data->dev_velocity[3 * v], 3);
    }

    template <typename Real>
    __global__ void calc_tetrahedron_force(ProjectiveDynamicsSolverData<Real> *data)
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
        Real F[9], P[9], H[9];
        cudaPhysics::matMul3(F, Ds, idm);
        cudaPhysics::calc_corotated_linear_P<Real>(P, F, data->m_lame_mu, data->m_lame_lambda);
        cudaPhysics::matmatTMul3(H, P, idm);
        cudaPhysics::vecMul(H, -data->dev_tet_volume[t], H, 9);
        Real f[12];
        cudaPhysics::matTrans3(&f[3], H);
        for (unsigned int j = 0; j < 3; ++j)
            f[j] = -(f[j + 3] + f[j + 6] + f[j + 9]);
        cudaPhysics::vecCopy(&data->dev_tet_force[12 * t], f, 12);
    }

    template <typename Real>
    __global__ void accumulate_vert_force(ProjectiveDynamicsSolverData<Real> *data)
    {
        unsigned int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= data->m_num_tet)
            return;
        for (unsigned int i = 0; i < 4; ++i)
        {
            unsigned int v = data->dev_tetrahedron[t * 4 + i];
            Real *f = &data->dev_tet_force[12 * t + 3 * i];
            for (unsigned int j = 0; j < 3; ++j)
            {
                atomicAdd(&data->dev_vert_force[3 * v + j], f[j]);
            }
        }
    }

    template <typename Real>
    __global__ void calc_ground_collision_force(ProjectiveDynamicsSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        if (data->dev_position[3 * v + 1] < 0.0f)
        {
            data->dev_vert_force[3 * v + 1] += -data->m_ground_collision_stiffness * data->dev_position[3 * v + 1];
            data->dev_constraint_Hessian_diag[v] += data->m_ground_collision_stiffness;
        }
    }

    template <typename Real>
    __global__ void jacobi_iteration(ProjectiveDynamicsSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        Real grad[3];
        Real delta_x[3];
        cudaPhysics::vecSubs3(delta_x, &data->dev_position_guess[3 * v], &data->dev_position[3 * v]);
        cudaPhysics::vecMul3(grad, data->dev_mass[v] * data->m_time_step_inv * data->m_time_step_inv, delta_x);
        cudaPhysics::vecAdd3(grad, grad, &data->dev_vert_force[3 * v]);
        Real B = data->dev_mass[v] * data->m_time_step_inv * data->m_time_step_inv - data->dev_stiffness_matrix_diag[v] + data->dev_constraint_Hessian_diag[v];
        cudaPhysics::axpby(&data->dev_position_next[3 * v], (Real)1, &data->dev_position[3 * v], (Real)1.0 / B, grad, 3);
    }

    template <typename Real>
    __global__ void Chebyshev(ProjectiveDynamicsSolverData<Real> *data, Real omega)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        Real delta_position[3];
        cudaPhysics::vecSubs3(delta_position, &data->dev_position_next[3 * v], &data->dev_position[3 * v]);
        cudaPhysics::axpby(&data->dev_position_next[3 * v], data->m_under_relaxation, delta_position, (Real)1.0, &data->dev_position[3 * v], 3);
        cudaPhysics::vecSubs3(delta_position, &data->dev_position_next[3 * v], &data->dev_position_prev[3 * v]);
        cudaPhysics::axpby(&data->dev_position_next[3 * v], omega, delta_position, (Real)1.0, &data->dev_position_prev[3 * v], 3);
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

    template <typename Real>
    __global__ void update_velocity(ProjectiveDynamicsSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        cudaPhysics::vecSubs3(&data->dev_velocity[3 * v], &data->dev_position[3 * v], &data->dev_position_backup[3 * v]);
        cudaPhysics::vecMul3(&data->dev_velocity[3 * v], data->m_time_step_inv, &data->dev_velocity[3 * v]);
    }
}

template <typename Real>
ProjectiveDynamicsSolver<Real>::ProjectiveDynamicsSolver(const std::vector<Real> &position, const std::vector<unsigned int> &tetrahedron, const std::vector<Real> &tetrahedron_density)
{
    m_data.m_num_vert = (unsigned int)position.size() / 3;
    m_data.m_num_tet = (unsigned int)tetrahedron.size() / 4;

    cudaMalloc((void **)&m_data.dev_position_backup, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_position_guess, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_position_prev, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemcpy(m_data.dev_position_prev, position.data(), sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_position, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemcpy(m_data.dev_position, position.data(), sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_position_next, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_velocity, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemset(m_data.dev_velocity, 0, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_mass, sizeof(Real) * m_data.m_num_vert);
    cudaMemset(m_data.dev_mass, 0, sizeof(Real) * m_data.m_num_vert);
    cudaMalloc((void **)&m_data.dev_vert_force, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_constraint_Hessian_diag, sizeof(Real) * m_data.m_num_vert);
    cudaMalloc((void **)&m_data.dev_stiffness_matrix_diag, sizeof(Real) * m_data.m_num_vert);
    cudaMemset(m_data.dev_stiffness_matrix_diag, 0, sizeof(Real) * m_data.m_num_vert);

    cudaMalloc((void **)&m_data.dev_tetrahedron, sizeof(unsigned int) * tetrahedron.size());
    cudaMemcpy(m_data.dev_tetrahedron, tetrahedron.data(), sizeof(unsigned int) * tetrahedron.size(), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_tet_density, sizeof(Real) * m_data.m_num_tet);
    cudaMemcpy(m_data.dev_tet_density, tetrahedron_density.data(), sizeof(Real) * m_data.m_num_tet, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_tet_volume, sizeof(Real) * m_data.m_num_tet);
    cudaMemset(m_data.dev_tet_volume, 0, sizeof(Real) * m_data.m_num_tet);
    cudaMalloc((void **)&m_data.dev_tet_force, sizeof(Real) * m_data.m_num_tet * 12);
    cudaMemset(m_data.dev_tet_force, 0, sizeof(Real) * m_data.m_num_tet * 12);
    cudaMalloc((void **)&m_data.dev_invDm, sizeof(Real) * m_data.m_num_tet * 9);

    m_data.m_time_step = TIME_STEP;
    m_data.m_time_step_inv = 1.0f / m_data.m_time_step;
    m_data.m_lame_mu = LAME_MU;
    m_data.m_lame_lambda = LAME_LAMBDA;
    m_data.m_ground_collision_stiffness = GROUND_COLLISION_STIFFNESS;
    m_data.m_under_relaxation = UNDER_RELAXATION;
    cudaMalloc(&m_data.dev_gravity, sizeof(Real) * 3);
    std::vector<Real> gravity = {0.0f, -GRAVITY, 0.0f};
    cudaMemcpy(m_data.dev_gravity, gravity.data(), sizeof(Real) * 3, cudaMemcpyHostToDevice);

    cudaMalloc(&m_dev_data, sizeof(ProjectiveDynamicsSolverData<Real>));
    cudaMemcpy(m_dev_data, &m_data, sizeof(ProjectiveDynamicsSolverData<Real>), cudaMemcpyHostToDevice);

    ProjectiveDynamicsSolverKernel::tetrahedron_initialize<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
    ProjectiveDynamicsSolverKernel::tet_stiffness_matrix_diag<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
}

template <typename Real>
ProjectiveDynamicsSolver<Real>::~ProjectiveDynamicsSolver()
{
    cudaFree(m_data.dev_position_backup);
    cudaFree(m_data.dev_position_guess);
    cudaFree(m_data.dev_position_prev);
    cudaFree(m_data.dev_position);
    cudaFree(m_data.dev_position_next);
    cudaFree(m_data.dev_velocity);
    cudaFree(m_data.dev_mass);
    cudaFree(m_data.dev_vert_force);
    cudaFree(m_data.dev_constraint_Hessian_diag);
    cudaFree(m_data.dev_stiffness_matrix_diag);

    cudaFree(m_data.dev_tetrahedron);
    cudaFree(m_data.dev_tet_density);
    cudaFree(m_data.dev_tet_volume);
    cudaFree(m_data.dev_tet_force);
    cudaFree(m_data.dev_invDm);

    cudaFree(m_data.dev_gravity);

    cudaFree(m_dev_data);
}

template <typename Real>
void ProjectiveDynamicsSolver<Real>::Step()
{
    cudaMemcpy(m_data.dev_position_backup, m_data.dev_position, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);
    ProjectiveDynamicsSolverKernel::initial_guess<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaMemcpy(m_data.dev_position_guess, m_data.dev_position, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);

    Real omega = 1;
    for (unsigned int iter = 0; iter < MAX_ITERATIONS; ++iter)
    {
        cudaMemset(m_data.dev_vert_force, 0, sizeof(Real) * m_data.m_num_vert * 3);
        cudaMemset(m_data.dev_constraint_Hessian_diag, 0, sizeof(Real) * m_data.m_num_vert);
        ProjectiveDynamicsSolverKernel::calc_tetrahedron_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
        ProjectiveDynamicsSolverKernel::accumulate_vert_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
        ProjectiveDynamicsSolverKernel::calc_ground_collision_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
        ProjectiveDynamicsSolverKernel::jacobi_iteration<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
        UpdateChebyshevOmega(omega, iter);
        ProjectiveDynamicsSolverKernel::Chebyshev<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data, omega);
        SwapPositionBuffers();
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

template <typename Real>
void ProjectiveDynamicsSolver<Real>::SwapPositionBuffers()
{
    std::swap(m_data.dev_position, m_data.dev_position_prev);
    std::swap(m_data.dev_position, m_data.dev_position_next);
    ProjectiveDynamicsSolverKernel::swap_position<Real><<<1, 1>>>(m_dev_data);
}

template struct ProjectiveDynamicsSolverData<float>;
template struct ProjectiveDynamicsSolverData<double>;
template class ProjectiveDynamicsSolver<float>;
template class ProjectiveDynamicsSolver<double>;

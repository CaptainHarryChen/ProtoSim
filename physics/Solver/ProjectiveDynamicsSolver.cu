#include "ProjectiveDynamicsSolver.cuh"
#include <cuda_utils/array.cuh>
#include <cuda_utils/block_size.cuh>
#include <cuda_utils/error.cuh>
#include <Math/algebra.cuh>

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
            ind[i] = data->dev_vert_to_tet[t * 4 + i] * 3;
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
        data->dev_tet_volume[t] = abs(vol) / 6.0;
        cudaPhysics::matInv3(&data->dev_invDm[t * 9 + 0], Dm);
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
        Real H[4];

        // generated code
        Real invDm0 = invDm[0], invDm1 = invDm[1], invDm2 = invDm[2], invDm3 = invDm[3],
             invDm4 = invDm[4], invDm5 = invDm[5], invDm6 = invDm[6], invDm7 = invDm[7], invDm8 = invDm[8];
        Real t2 = invDm0 * invDm0;
        Real t3 = invDm1 * invDm1;
        Real t4 = invDm2 * invDm2;
        Real t5 = invDm3 * invDm3;
        Real t6 = invDm4 * invDm4;
        Real t7 = invDm5 * invDm5;
        Real t8 = invDm6 * invDm6;
        Real t9 = invDm7 * invDm7;
        Real t10 = invDm8 * invDm8;
        Real t11 = invDm0 + invDm3 + invDm6;
        Real t12 = invDm1 + invDm4 + invDm7;
        Real t13 = invDm2 + invDm5 + invDm8;
        Real t14 = t2 * 2.0;
        Real t15 = t3 * 2.0;
        Real t16 = t4 * 2.0;
        Real t17 = t5 * 2.0;
        Real t18 = t6 * 2.0;
        Real t19 = t7 * 2.0;
        Real t20 = t8 * 2.0;
        Real t21 = t9 * 2.0;
        Real t22 = t10 * 2.0;
        Real t23 = t11 * t11;
        Real t24 = t12 * t12;
        Real t25 = t13 * t13;
        Real t26 = t23 * 2.0;
        Real t27 = t24 * 2.0;
        Real t28 = t25 * 2.0;
        Real t29 = t14 + t15 + t16;
        Real t30 = t17 + t18 + t19;
        Real t31 = t20 + t21 + t22;
        Real t32 = t26 + t27 + t28;
        H[0] = t32;
        H[1] = t29;
        H[2] = t30;
        H[3] = t31;

        Real rate = data->dev_tet_volume[t] * data->m_stiffness;
        for (unsigned int i = 0; i < 4; ++i)
            data->dev_diag_Hessian[v[i]] = H[i];
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
    cudaMalloc((void **)&m_data.dev_position_next, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_position_backup, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_position_delta, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_position_delta_denominator, sizeof(Real) * m_data.m_num_vert);
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
    cudaMalloc((void **)&m_data.dev_invDm, sizeof(Real) * m_data.m_num_tet * 9);

    cudaMalloc((void **)&m_data.dev_energy, sizeof(Real));

    cudaMalloc((void **)&m_data.dev_ground_collision_count, sizeof(unsigned int));
    cudaMalloc((void **)&m_data.dev_ground_collision_ids, sizeof(unsigned int) * m_data.m_num_vert);

    m_data.m_time_step = TIME_STEP;
    m_data.m_lame_mu = LAME_MU;
    m_data.m_lame_lambda = LAME_LAMBDA;
    m_data.m_stiffness = 18000000 * 0.5;
    m_data.m_collision_stiffness = 1000000;
    cudaMalloc(&m_data.dev_gravity, sizeof(Real) * 3);
    std::vector<Real> gravity = {0., -GRAVITY, 0.};
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
    cudaFree(m_data.dev_position_delta_denominator);
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

    cudaFree(m_data.dev_energy);

    cudaFree(m_data.dev_ground_collision_count);
    cudaFree(m_data.dev_ground_collision_ids);

    cudaFree(m_data.dev_gravity);
    
    cudaFree(m_dev_data);
}

template <typename Real>
void ProjectiveDynamicsSolver<Real>::Step()
{
    // cudaMemcpy(this->dev_position_backup, this->dev_position, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);

    // initial_guess_kernel<Real> << < this->blocksPerGrid_vert, THREADS_PER_BLOCK >> > (m_data.m_num_vert, this->m_time_step, this->dev_gravity, this->dev_position, this->dev_velocities, dev_constrained);
    // cudaMemcpy(this->dev_inertia, this->dev_position, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);
    // compute_mass_matrix_kernel<Real> << < this->blocksPerGrid_vert, THREADS_PER_BLOCK >> > (m_data.m_num_vert, this->dev_position, 1. / this->m_time_step, this->dev_mass, dev_init_A, dev_init_B);

    // Real omega;
    // Real rho = 0.9992;
    // unsigned int max_iter = 64;
    // for (unsigned int iter = 0; iter < max_iter; ++iter)
    // {
    //     projective_dynamics_constraint_kernel<Real> << <this->blocksPerGrid_tet, THREADS_PER_BLOCK >> > (m_data.m_num_tet, this->dev_tetrahedras, m_stiffness, this->dev_position, this->dev_tet_volume, this->dev_invDm, dev_tet_force);

    //     cudaMemset(this->dev_ground_collision_count, 0, sizeof(unsigned int));
    //     ground_collision_detection<Real> << <this->blocksPerGrid_vert, THREADS_PER_BLOCK >> > (m_data.m_num_vert, this->dev_position, this->dev_ground_collision_count, this->dev_ground_collision_ids);
    //     this->m_ground_collision_count = visit_device(this->dev_ground_collision_count, 0);
    //     cudaMemset(this->dev_vert_force, 0, sizeof(Real) * m_data.m_num_vert * 3);
    //     cudaMemset(this->dev_vert_Hessian, 0, sizeof(Real) * m_data.m_num_vert);
    //     compute_ground_collision_force<Real> << <get_blocksPerGrid(this->m_ground_collision_count), THREADS_PER_BLOCK >> > (this->m_ground_collision_count, this->dev_position, this->m_collision_stiffness, this->dev_vert_force, this->dev_vert_Hessian, this->dev_ground_collision_ids);

    //     //compute_iteration_kernel<Real> << <this->blocksPerGrid_vert, THREADS_PER_BLOCK >> > (m_data.m_num_vert, this->dev_position, dev_position_next, dev_init_A, dev_init_B, dev_diag_Hessian, dev_vert_to_tet, dev_vert_to_tet_offset,
    //     //    this->dev_tetrahedras, dev_tet_force, dev_constrained);

    //     compute_iteration_kernel<Real> << <this->blocksPerGrid_vert, THREADS_PER_BLOCK >> > (m_data.m_num_vert, this->dev_position, dev_position_next, dev_init_A, dev_init_B, dev_diag_Hessian, dev_vert_to_tet, dev_vert_to_tet_offset,
    //         this->dev_tetrahedras, dev_tet_force, this->dev_vert_force, this->dev_vert_Hessian, this->dev_position_delta, dev_constrained);
    //     Real alpha = 1.;
    //     if (iter % 8 == 0)
    //     {
    //         alpha = this->line_searches();
    //     }
    //     //Real alpha = this->line_searches();
    //     update_position_kernel<Real> << <this->blocksPerGrid_vert, THREADS_PER_BLOCK >> > (m_data.m_num_vert, this->dev_position_next, this->dev_position, this->dev_position_delta, alpha);

    //     if (iter <= 10) omega = 1;
    //     else if (iter == 11) omega = 2 / (2 - rho * rho);
    //     else omega = 4 / (4 - rho * rho * omega);

    //     Chebyshev_kernel<Real> << <this->blocksPerGrid_vert, THREADS_PER_BLOCK >> > (m_data.m_num_vert, this->dev_position_prev, this->dev_position, this->dev_position_next, omega);

    //     swap<Real*>(this->dev_position, this->dev_position_prev);
    //     swap<Real*>(this->dev_position, this->dev_position_next);
    // }
    // update_velocity_kernel<Real> << <this->blocksPerGrid_vert, THREADS_PER_BLOCK >> > (m_data.m_num_vert, dev_position_backup, this->dev_position, 1. / this->m_time_step, this->dev_velocities);
}

template <typename Real>
Real *ProjectiveDynamicsSolver<Real>::GetDevicePositions()
{
    return this->m_data.dev_position;
}

template struct ProjectiveDynamicsSolverData<float>;
template struct ProjectiveDynamicsSolverData<double>;
template class ProjectiveDynamicsSolver<float>;
template class ProjectiveDynamicsSolver<double>;

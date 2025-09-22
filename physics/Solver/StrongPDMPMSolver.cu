#include "StrongPDMPMSolver.cuh"
#include <thrust/fill.h>
#include <cuda_utils/cuda_utils.cuh>
#include <Math/algebra.cuh>
#include <Math/elastic_model.cuh>

namespace StrongPDMPMSolverKernel
{
    template <typename Real>
    __global__ void tetrahedron_initialize(StrongPDMPMSolverData<Real> *data)
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
    __global__ void tet_stiffness_matrix_diag(StrongPDMPMSolverData<Real> *data)
    {
        unsigned int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= data->m_num_tet)
            return;
        unsigned int *v = &data->dev_tetrahedron[4 * t];
        Real *invDm = &data->dev_invDm[9 * t];
        Real delta_f[9]; // delta_f = -2 * V0 * stiffness * invDm * invDm^T * (delta Ds^T)
        cudaPhysics::matmatTMul3(delta_f, invDm, invDm);
        cudaPhysics::vecMul(delta_f, -2 * data->dev_tet_volume[t] * data->m_lame_mu, delta_f, 9);
        // stiffness matrix K are 12x12, which has 12 diagonal elements. But every 3 elements are the same. So we only need 4 elements.
        Real K_diag[4];
        K_diag[1] = delta_f[0]; //(delta Ds^T)[0][0:3] = 1
        K_diag[2] = delta_f[4]; //(delta Ds^T)[1][0:3] = 1
        K_diag[3] = delta_f[8]; //(delta Ds^T)[2][0:3] = 1
        // (delta Ds^T)[:,:] = -1 and f[0] = - f[1] - f[2] - f[3] which means delta_f[0] need a sum
        K_diag[0] = delta_f[0] + delta_f[1] + delta_f[2] + delta_f[3] + delta_f[4] + delta_f[5] + delta_f[6] + delta_f[7] + delta_f[8];
        for (unsigned int i = 0; i < 4; ++i)
            atomicAdd(&data->dev_stiffness_matrix_diag[v[i]], K_diag[i]);
    }

    template <typename Real>
    __global__ void sample_initialize(StrongPDMPMSolverData<Real> *data)
    {
        unsigned int s = blockDim.x * blockIdx.x + threadIdx.x;
        if (s >= data->m_num_sample)
            return;
        unsigned int tri_idx = (unsigned int)data->dev_sample_tri_idx[s];
        Real mass_inv = 0.0f;
        for (unsigned int i = 0; i < 3; ++i)
        {
            unsigned int v = data->dev_triangle[tri_idx * 3 + i];
            mass_inv += (data->dev_sample_barycentric[s * 3 + i] * data->dev_sample_barycentric[s * 3 + i]) / data->dev_mass[v];
        }
        data->dev_sample_mass[s] = 1.0f / mass_inv;
        data->dev_sample_J[s] = 1.0f;
    }

    template <typename Real>
    __global__ void initial_guess(StrongPDMPMSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        cudaPhysics::axpby(&data->dev_velocity[3 * v], (Real)1.0, &data->dev_velocity[3 * v], data->m_time_step, data->dev_gravity, 3);
        cudaPhysics::axpby(&data->dev_position[3 * v], (Real)1.0, &data->dev_position[3 * v], data->m_time_step, &data->dev_velocity[3 * v], 3);
    }

    template <typename Real>
    __global__ void calc_tetrahedron_force(StrongPDMPMSolverData<Real> *data)
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
        cudaPhysics::polar_decomposition_R(R, F);

        Real f[9]; // f = -2 * V0 * stiffness * invDm * (F - R)^T
        Real FSubR[9];
        cudaPhysics::vecSubs(FSubR, F, R, 9);
        cudaPhysics::matmatTMul3(f, idm, FSubR);
        cudaPhysics::vecMul(f, -2 * data->dev_tet_volume[t] * data->m_lame_mu, f, 9);
        cudaPhysics::vecCopy(&data->dev_tet_force[12 * t + 3], f, 9);
        cudaPhysics::axpbypcz(&data->dev_tet_force[12 * t], (Real)-1.0, &f[0], (Real)-1.0, &f[3], (Real)-1.0, &f[6], 3);
    }

    template <typename Real>
    __global__ void accumulate_vert_force(StrongPDMPMSolverData<Real> *data)
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
    __global__ void calc_ground_collision_force(StrongPDMPMSolverData<Real> *data)
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
    __global__ void jacobi_iteration(StrongPDMPMSolverData<Real> *data)
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
    __global__ void Chebyshev(StrongPDMPMSolverData<Real> *data, Real omega)
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
    __global__ void swap_position(StrongPDMPMSolverData<Real> *data)
    {
        Real *temp = data->dev_position;
        data->dev_position = data->dev_position_prev;
        data->dev_position_prev = temp;
        temp = data->dev_position;
        data->dev_position = data->dev_position_next;
        data->dev_position_next = temp;
    }

    template <typename Real>
    __global__ void update_velocity(StrongPDMPMSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        cudaPhysics::vecSubs3(&data->dev_velocity[3 * v], &data->dev_position[3 * v], &data->dev_position_backup[3 * v]);
        cudaPhysics::vecMul3(&data->dev_velocity[3 * v], data->m_time_step_inv, &data->dev_velocity[3 * v]);
    }

    template <typename Real>
    __global__ void calc_sample_position(StrongPDMPMSolverData<Real> *data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data->m_num_sample)
            return;
        unsigned int tri_idx = (unsigned int)data->dev_sample_tri_idx[i];
        Real barycentric[3];
        barycentric[0] = data->dev_sample_barycentric[i * 3 + 0];
        barycentric[1] = data->dev_sample_barycentric[i * 3 + 1];
        barycentric[2] = data->dev_sample_barycentric[i * 3 + 2];
        unsigned int v0 = data->dev_triangle[tri_idx * 3 + 0] * 3;
        unsigned int v1 = data->dev_triangle[tri_idx * 3 + 1] * 3;
        unsigned int v2 = data->dev_triangle[tri_idx * 3 + 2] * 3;
        Real *sample_position = &data->dev_sample_position[i * 3];
        sample_position[0] = barycentric[0] * data->dev_position[v0 + 0] + barycentric[1] * data->dev_position[v1 + 0] + barycentric[2] * data->dev_position[v2 + 0];
        sample_position[1] = barycentric[0] * data->dev_position[v0 + 1] + barycentric[1] * data->dev_position[v1 + 1] + barycentric[2] * data->dev_position[v2 + 1];
        sample_position[2] = barycentric[0] * data->dev_position[v0 + 2] + barycentric[1] * data->dev_position[v1 + 2] + barycentric[2] * data->dev_position[v2 + 2];
    }
}

template <typename Real>
StrongPDMPMSolver<Real>::StrongPDMPMSolver(
    const std::vector<Real> &node_position,
    const std::vector<unsigned int> &surface_triangle,
    const std::vector<unsigned int> &tetrahedron, 
    const std::vector<Real> &tetrahedron_density,

    const std::vector<Real> &sample_barycentric_weights,
    const std::vector<unsigned int> &sample_triangle_idx,
    const std::vector<Real> &sample_area,

    std::vector<Real> bbox,
    Real grid_spacing,
    unsigned int boundary_thickness
)
{
    m_data.m_num_vert = (unsigned int)node_position.size() / 3;
    m_data.m_num_tet = (unsigned int)tetrahedron.size() / 4;
    m_data.m_num_tri = (unsigned int)surface_triangle.size() / 3;
    m_data.m_num_sample = (unsigned int)sample_barycentric_weights.size() / 3;
    assert(m_data.m_num_sample == sample_triangle_idx.size());
    printf("num_vert = %u, num_tet = %u, num_tri = %u, num_sample = %u\n", m_data.m_num_vert, m_data.m_num_tet, m_data.m_num_tri, m_data.m_num_sample);

    cudaMalloc((void **)&m_data.dev_position_backup, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_position_guess, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_position_prev, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemcpy(m_data.dev_position_prev, node_position.data(), sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_position, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemcpy(m_data.dev_position, node_position.data(), sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_position_next, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_velocity, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemset(m_data.dev_velocity, 0, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_mass, sizeof(Real) * m_data.m_num_vert);
    cudaMemset(m_data.dev_mass, 0, sizeof(Real) * m_data.m_num_vert);
    cudaMalloc((void **)&m_data.dev_vert_force, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_constraint_Hessian_diag, sizeof(Real) * m_data.m_num_vert);
    cudaMalloc((void **)&m_data.dev_stiffness_matrix_diag, sizeof(Real) * m_data.m_num_vert);
    cudaMemset(m_data.dev_stiffness_matrix_diag, 0, sizeof(Real) * m_data.m_num_vert);

    cudaMalloc((void **)&m_data.dev_tetrahedron, sizeof(unsigned int) * m_data.m_num_tet * 4);
    cudaMemcpy(m_data.dev_tetrahedron, tetrahedron.data(), sizeof(unsigned int) * m_data.m_num_tet * 4, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_tet_density, sizeof(Real) * m_data.m_num_tet);
    cudaMemcpy(m_data.dev_tet_density, tetrahedron_density.data(), sizeof(Real) * m_data.m_num_tet, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_tet_volume, sizeof(Real) * m_data.m_num_tet);
    cudaMemset(m_data.dev_tet_volume, 0, sizeof(Real) * m_data.m_num_tet);
    cudaMalloc((void **)&m_data.dev_tet_force, sizeof(Real) * m_data.m_num_tet * 12);
    cudaMemset(m_data.dev_tet_force, 0, sizeof(Real) * m_data.m_num_tet * 12);
    cudaMalloc((void **)&m_data.dev_invDm, sizeof(Real) * m_data.m_num_tet * 9);

    cudaMalloc((void **)&m_data.dev_triangle, sizeof(unsigned int) * m_data.m_num_tri * 3);
    cudaMemcpy(m_data.dev_triangle, surface_triangle.data(), sizeof(unsigned int) * m_data.m_num_tri * 3, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&m_data.dev_sample_position, sizeof(Real) * m_data.m_num_sample * 3);
    cudaMalloc((void **)&m_data.dev_sample_barycentric, sizeof(Real) * m_data.m_num_sample * 3);
    cudaMemcpy(m_data.dev_sample_barycentric, sample_barycentric_weights.data(), sizeof(Real) * m_data.m_num_sample * 3, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_sample_tri_idx, sizeof(unsigned int) * m_data.m_num_sample);
    cudaMemcpy(m_data.dev_sample_tri_idx, sample_triangle_idx.data(), sizeof(unsigned int) * m_data.m_num_sample, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_sample_area, sizeof(Real) * m_data.m_num_sample);
    cudaMemcpy(m_data.dev_sample_area, sample_area.data(), sizeof(Real) * m_data.m_num_sample, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_sample_velocity, sizeof(Real) * m_data.m_num_sample * 3);
    cudaMalloc((void **)&m_data.dev_sample_mass, sizeof(Real) * m_data.m_num_sample);
    cudaMalloc((void **)&m_data.dev_sample_J, sizeof(Real) * m_data.m_num_sample);
    cudaMalloc((void **)&m_data.dev_sample_temp_J, sizeof(Real) * m_data.m_num_sample);

    // need expand the bbox to ensure the particles can get 3x3x3 grids
    std::vector<Real> inner_bbox;
    std::vector<Real> outer_bbox;
    std::vector<unsigned int> grid_size; // order: x y z
    m_data.m_grid_spacing = grid_spacing;
    m_data.m_boundary_thickness = boundary_thickness;
    inner_bbox = bbox;
    outer_bbox = bbox;
    for (unsigned int i = 0; i < 3; ++i)
    {
        outer_bbox[i] -= m_data.m_boundary_thickness * m_data.m_grid_spacing;
        outer_bbox[i + 3] += m_data.m_boundary_thickness * m_data.m_grid_spacing;
    }
    grid_size.clear();
    for (unsigned int i = 0; i < 3; ++i)
        grid_size.push_back((unsigned int)(floor((outer_bbox[i + 3] - outer_bbox[i]) / m_data.m_grid_spacing + 0.5)) + 1);
    for (unsigned int i = 0; i < 3; ++i)
        outer_bbox[i + 3] = outer_bbox[i] + m_data.m_grid_spacing * grid_size[i];
    m_data.m_num_grid = grid_size[0] * grid_size[1] * grid_size[2];
    printf("num_grid = %u grid_size = [%u %u %u] boundary_thickness = %u\n", m_data.m_num_grid, grid_size[0], grid_size[1], grid_size[2], m_data.m_boundary_thickness);

    cudaMalloc(&m_data.dev_inner_bbox, sizeof(Real) * 6);
    cudaMemcpy(m_data.dev_inner_bbox, inner_bbox.data(), sizeof(Real) * 6, cudaMemcpyHostToDevice);
    cudaMalloc(&m_data.dev_outer_bbox, sizeof(Real) * 6);
    cudaMemcpy(m_data.dev_outer_bbox, outer_bbox.data(), sizeof(Real) * 6, cudaMemcpyHostToDevice);
    cudaMalloc(&m_data.dev_grid_size, sizeof(unsigned int) * 3);
    cudaMemcpy(m_data.dev_grid_size, grid_size.data(), sizeof(unsigned int) * 3, cudaMemcpyHostToDevice);
    cudaMalloc(&m_data.dev_grid_momentum, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_momentum, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_mass, sizeof(Real) * m_data.m_num_grid);
    cudaMemset(m_data.dev_grid_mass, 0, sizeof(Real) * m_data.m_num_grid);
    cudaMalloc(&m_data.dev_grid_velocity, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_velocity, 0, sizeof(Real) * m_data.m_num_grid * 3);

    m_data.m_time_step = TIME_STEP;
    m_data.m_time_step_inv = 1.0f / m_data.m_time_step;
    m_data.m_lame_mu = LAME_MU;
    m_data.m_lame_lambda = LAME_LAMBDA;
    m_data.m_ground_collision_stiffness = GROUND_COLLISION_STIFFNESS;
    m_data.m_under_relaxation = UNDER_RELAXATION;
    cudaMalloc(&m_data.dev_gravity, sizeof(Real) * 3);
    std::vector<Real> gravity = {0.0f, -GRAVITY, 0.0f};
    cudaMemcpy(m_data.dev_gravity, gravity.data(), sizeof(Real) * 3, cudaMemcpyHostToDevice);

    cudaMalloc(&m_dev_data, sizeof(StrongPDMPMSolverData<Real>));
    cudaMemcpy(m_dev_data, &m_data, sizeof(StrongPDMPMSolverData<Real>), cudaMemcpyHostToDevice);

    StrongPDMPMSolverKernel::tetrahedron_initialize<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
    StrongPDMPMSolverKernel::tet_stiffness_matrix_diag<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
    StrongPDMPMSolverKernel::calc_sample_position<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
    StrongPDMPMSolverKernel::sample_initialize<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
}

template <typename Real>
StrongPDMPMSolver<Real>::~StrongPDMPMSolver()
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

    cudaFree(m_data.dev_triangle);

    cudaFree(m_data.dev_sample_position);
    cudaFree(m_data.dev_sample_barycentric);
    cudaFree(m_data.dev_sample_tri_idx);
    cudaFree(m_data.dev_sample_area);
    cudaFree(m_data.dev_sample_velocity);
    cudaFree(m_data.dev_sample_mass);
    cudaFree(m_data.dev_sample_J);
    cudaFree(m_data.dev_sample_temp_J);

    cudaFree(m_data.dev_inner_bbox);
    cudaFree(m_data.dev_outer_bbox);
    cudaFree(m_data.dev_grid_size);
    cudaFree(m_data.dev_grid_momentum);
    cudaFree(m_data.dev_grid_mass);
    cudaFree(m_data.dev_grid_velocity);

    cudaFree(m_data.dev_gravity);

    cudaFree(m_dev_data);
}

template <typename Real>
void StrongPDMPMSolver<Real>::Step()
{
    cudaMemcpy(m_data.dev_position_backup, m_data.dev_position, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);
    StrongPDMPMSolverKernel::initial_guess<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaMemcpy(m_data.dev_position_guess, m_data.dev_position, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);

    Real omega = 1;
    for (unsigned int iter = 0; iter < MAX_ITERATIONS; ++iter)
    {
        cudaMemset(m_data.dev_vert_force, 0, sizeof(Real) * m_data.m_num_vert * 3);
        cudaMemset(m_data.dev_constraint_Hessian_diag, 0, sizeof(Real) * m_data.m_num_vert);
        StrongPDMPMSolverKernel::calc_tetrahedron_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
        StrongPDMPMSolverKernel::accumulate_vert_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
        StrongPDMPMSolverKernel::calc_ground_collision_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
        StrongPDMPMSolverKernel::jacobi_iteration<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
        UpdateChebyshevOmega(omega, iter);
        StrongPDMPMSolverKernel::Chebyshev<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data, omega);
        SwapPositionBuffers();
    }
    StrongPDMPMSolverKernel::update_velocity<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
    StrongPDMPMSolverKernel::calc_sample_position<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
}

template <typename Real>
Real *StrongPDMPMSolver<Real>::GetDeviceNodePositions()
{
    return this->m_data.dev_position;
}

template <typename Real>
Real *StrongPDMPMSolver<Real>::GetDeviceSamplePositions()
{
    return this->m_data.dev_sample_position;
}

template <typename Real>
void StrongPDMPMSolver<Real>::UpdateChebyshevOmega(Real &omega, unsigned int iter)
{
    if (iter < CHEBYSHEV_DELAY_ITER)
        omega = 1;
    else if (iter == CHEBYSHEV_DELAY_ITER)
        omega = 2 / (2 - CHEBYSHEV_RHO * CHEBYSHEV_RHO);
    else
        omega = 4 / (4 - CHEBYSHEV_RHO * CHEBYSHEV_RHO * omega);
}

template <typename Real>
void StrongPDMPMSolver<Real>::SwapPositionBuffers()
{
    std::swap(m_data.dev_position, m_data.dev_position_prev);
    std::swap(m_data.dev_position, m_data.dev_position_next);
    StrongPDMPMSolverKernel::swap_position<Real><<<1, 1>>>(m_dev_data);
}

template struct StrongPDMPMSolverData<float>;
template struct StrongPDMPMSolverData<double>;
template class StrongPDMPMSolver<float>;
template class StrongPDMPMSolver<double>;

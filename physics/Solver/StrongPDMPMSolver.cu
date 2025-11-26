#include "StrongPDMPMSolver.cuh"
#include <thrust/fill.h>
#include <cuda_utils/cuda_utils.cuh>
#include <Math/algebra.cuh>
#include <Math/ConstitutiveModel/Neohookean.cuh>

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
    __global__ void calc_triangle_normal(StrongPDMPMSolverData<Real> *data)
    {
        unsigned int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= data->m_num_tri)
            return;
        unsigned int v0 = data->dev_triangle[t * 3 + 0] * 3;
        unsigned int v1 = data->dev_triangle[t * 3 + 1] * 3;
        unsigned int v2 = data->dev_triangle[t * 3 + 2] * 3;
        Real e1[3], e2[3];
        cudaPhysics::vecSubs3(e1, &data->dev_position[v1], &data->dev_position[v0]);
        cudaPhysics::vecSubs3(e2, &data->dev_position[v2], &data->dev_position[v0]);
        cudaPhysics::cross3(&data->dev_tri_normal[t * 3], e1, e2);
        cudaPhysics::norm3InPlace(&data->dev_tri_normal[t * 3]);
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
    __global__ void calc_sample_to_leftbottom_grid_id(StrongPDMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_sample)
            return;
        // find the nearest grid, so plus 0.5 and floor
        int x = floor((data->dev_sample_position[i * 3 + 0] - data->dev_outer_bbox[0]) / data->m_grid_spacing + 0.5);
        int y = floor((data->dev_sample_position[i * 3 + 1] - data->dev_outer_bbox[1]) / data->m_grid_spacing + 0.5);
        int z = floor((data->dev_sample_position[i * 3 + 2] - data->dev_outer_bbox[2]) / data->m_grid_spacing + 0.5);
        // get the left bottom grid id
        x--;
        y--;
        z--;
        if (x < 0 || y < 0 || z < 0 || x >= data->dev_grid_size[0] - 2 || y >= data->dev_grid_size[1] - 2 || z >= data->dev_grid_size[2] - 2)
        {
            x = max(0, min(x, (int)data->dev_grid_size[0] - 3));
            y = max(0, min(y, (int)data->dev_grid_size[1] - 3));
            z = max(0, min(z, (int)data->dev_grid_size[2] - 3));
            printf("Warning: sample %d is out of grid, set to %d %d %d [Sample position] %.10f %.10f %.10f\n", i, x, y, z, data->dev_sample_position[i * 3 + 0], data->dev_sample_position[i * 3 + 1], data->dev_sample_position[i * 3 + 2]);
        }

        data->dev_sample_to_grid_id[i] = x * data->dev_grid_size[1] * data->dev_grid_size[2] + y * data->dev_grid_size[2] + z;
    }

    template <typename Real>
    __device__ void get_grid_position(Real *grid_position, unsigned int id, StrongPDMPMSolverData<Real> *data)
    {
        unsigned int z = id % data->dev_grid_size[2];
        unsigned int y = (id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int x = id / data->dev_grid_size[2] / data->dev_grid_size[1];
        grid_position[0] = data->dev_outer_bbox[0] + x * data->m_grid_spacing;
        grid_position[1] = data->dev_outer_bbox[1] + y * data->m_grid_spacing;
        grid_position[2] = data->dev_outer_bbox[2] + z * data->m_grid_spacing;
    }

    template <typename Real>
    __device__ Real grid_particle_quadratic_weight(const Real *grid_position, const Real *particle_position, Real grid_spacing)
    {
        Real result = 1.;
        for (unsigned int i = 0; i < 3; ++i)
        {
            Real d = (particle_position[i] - grid_position[i]) / grid_spacing;
            Real w = 0;
            if (-0.5 < d && d < 0.5)
                w = 0.75 - d * d;
            else if (0.5 <= d && d < 1.5)
                w = 0.5 * (1.5 - d) * (1.5 - d);
            else if (-1.5 < d && d <= -0.5)
                w = 0.5 * (1.5 + d) * (1.5 + d);
            result *= w;
        }
        return result;
    }

    template <typename Real>
    __global__ void P2G_mass(StrongPDMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_sample)
            return;
        const Real *sample_position = &data->dev_sample_position[i * 3];
        const Real *sample_velocity = &data->dev_sample_velocity[i * 3];
        unsigned int leftbottom_grid_id = data->dev_sample_to_grid_id[i];
        unsigned int x = leftbottom_grid_id / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (leftbottom_grid_id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = leftbottom_grid_id % data->dev_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data->dev_grid_size[1] * data->dev_grid_size[2] + (y + dy) * data->dev_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    get_grid_position(grid_position, grid_id, data);
                    Real weight = grid_particle_quadratic_weight(grid_position, sample_position, data->m_grid_spacing);
                    atomicAdd(&data->dev_grid_mass[grid_id], weight * data->dev_sample_mass[i]);
                }
    }

    template <typename Real>
    __global__ void P2G_grid_diag_K(StrongPDMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_sample)
            return;
        const Real *sample_position = &data->dev_sample_position[i * 3];
        unsigned int leftbottom_grid_id = data->dev_sample_to_grid_id[i];
        unsigned int x = leftbottom_grid_id / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (leftbottom_grid_id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = leftbottom_grid_id % data->dev_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data->dev_grid_size[1] * data->dev_grid_size[2] + (y + dy) * data->dev_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    get_grid_position(grid_position, grid_id, data);
                    Real weight = grid_particle_quadratic_weight(grid_position, sample_position, data->m_grid_spacing);
                    Real delta_position[3];
                    cudaPhysics::vecSubs3(delta_position, grid_position, sample_position);
                    
                    Real diag_K[3];
                    cudaPhysics::vecMul3(diag_K, delta_position, delta_position);
                    cudaPhysics::vecMul3(diag_K, -data->dev_sample_volume[i]
                                                * data->m_lame_lambda
                                                * data->dev_sample_J[i] * data->dev_sample_J[i]
                                                * 16 * weight * weight
                                                / (data->m_grid_spacing * data->m_grid_spacing * data->m_grid_spacing * data->m_grid_spacing));
                    atomicAdd(&data->dev_grid_diag_K[grid_id * 3 + 0], diag_K[0]);
                    atomicAdd(&data->dev_grid_diag_K[grid_id * 3 + 1], diag_K[1]);
                    atomicAdd(&data->dev_grid_diag_K[grid_id * 3 + 2], diag_K[2]);
                }
    }

    template <typename Real>
    __global__ void G2P_sample_diag_K(StrongPDMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_sample)
            return;
        const Real *sample_position = &data->dev_sample_position[i * 3];
        const Real *sample_normal = &data->dev_tri_normal[data->dev_sample_tri_idx[i] * 3];
        Real sample_mass = data->dev_sample_mass[i];
        Real *sample_diag_K = &data->dev_sample_diag_K[i * 3];
        sample_diag_K[0] = sample_diag_K[1] = sample_diag_K[2] = 0;
        unsigned int leftbottom_grid_id = data->dev_sample_to_grid_id[i];
        unsigned int x = leftbottom_grid_id / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (leftbottom_grid_id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = leftbottom_grid_id % data->dev_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data->dev_grid_size[1] * data->dev_grid_size[2] + (y + dy) * data->dev_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    get_grid_position(grid_position, grid_id, data);
                    Real weight = grid_particle_quadratic_weight(grid_position, sample_position, data->m_grid_spacing);

                    cudaPhysics::axpby(sample_diag_K, (Real)1.0, sample_diag_K, weight * weight / data->dev_grid_mass[grid_id], &data->dev_grid_diag_K[grid_id * 3], 3);
                }
        cudaPhysics::vecMul3(sample_diag_K, sample_diag_K, sample_normal);
        cudaPhysics::vecMul3(sample_diag_K, sample_diag_K, sample_normal);
        cudaPhysics::vecMul3(sample_diag_K, sample_mass);
    }

    template <typename Real>
    __global__ void accumulate_collision_Hessian_diag_from_sample(StrongPDMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockIdx.x + threadIdx.x;
        if (i >= data->m_num_sample)
            return;
        Real *sample_diag_K = &data->dev_sample_diag_K[i * 3];
        for (unsigned int j = 0; j < 3; ++j)
        {
            Real weight = data->dev_sample_barycentric[i * 3 + j];
            unsigned int v = data->dev_triangle[data->dev_sample_tri_idx[i] * 3 + j];
            atomicAdd(&data->dev_collision_Hessian_diag[v * 3 + 0], sample_diag_K[0] * weight * weight * data->m_time_step_inv);
            atomicAdd(&data->dev_collision_Hessian_diag[v * 3 + 1], sample_diag_K[1] * weight * weight * data->m_time_step_inv);
            atomicAdd(&data->dev_collision_Hessian_diag[v * 3 + 2], sample_diag_K[2] * weight * weight * data->m_time_step_inv);
        }
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
    __global__ void accumulate_vert_force_from_tet(StrongPDMPMSolverData<Real> *data)
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
    __global__ void P2G_momentum(StrongPDMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_sample)
            return;
        const Real *sample_position = &data->dev_sample_position[i * 3];
        const Real *sample_velocity = &data->dev_sample_velocity[i * 3];
        unsigned int leftbottom_grid_id = data->dev_sample_to_grid_id[i];
        unsigned int x = leftbottom_grid_id / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (leftbottom_grid_id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = leftbottom_grid_id % data->dev_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data->dev_grid_size[1] * data->dev_grid_size[2] + (y + dy) * data->dev_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    get_grid_position(grid_position, grid_id, data);
                    Real weight = grid_particle_quadratic_weight(grid_position, sample_position, data->m_grid_spacing);
                    Real momentum[3] = {0, 0, 0};
                    Real delta_position[3];
                    cudaPhysics::vecSubs3(delta_position, grid_position, sample_position);
                    // cudaPhysics::matVec3(momentum, &data->dev_sample_C[i * 9], delta_position);
                    // cudaPhysics::vecMul3(momentum, data->dev_sample_mass[i], momentum);

                    Real temp_momentum[3];
                    cudaPhysics::vecMul3(temp_momentum, data->dev_sample_mass[i], sample_velocity);
                    cudaPhysics::vecAdd3(momentum, temp_momentum, momentum);
                    cudaPhysics::vecMul3(momentum, weight, momentum);
                    atomicAdd(&data->dev_grid_momentum[grid_id * 3 + 0], momentum[0]);
                    atomicAdd(&data->dev_grid_momentum[grid_id * 3 + 1], momentum[1]);
                    atomicAdd(&data->dev_grid_momentum[grid_id * 3 + 2], momentum[2]);
                }
    }

    template <typename Real>
    __global__ void calc_grids_velocity(StrongPDMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        if (data->dev_grid_mass[i] > 0)
            cudaPhysics::vecMul3(&data->dev_grid_velocity[i * 3], (Real)(1. / data->dev_grid_mass[i]), &data->dev_grid_momentum[i * 3]);
        else
        {
            data->dev_grid_velocity[i * 3 + 0] = 0;
            data->dev_grid_velocity[i * 3 + 1] = 0;
            data->dev_grid_velocity[i * 3 + 2] = 0;
        }
    }

    template <typename Real>
    __global__ void G2P_calc_sample_temp_J(StrongPDMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_sample)
            return;
        Real &temp_J = data->dev_sample_temp_J[i];
        temp_J = 0;
        const Real *sample_position = &data->dev_sample_position[i * 3];
        unsigned int leftbottom_grid_id = data->dev_sample_to_grid_id[i];
        unsigned int x = leftbottom_grid_id / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (leftbottom_grid_id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = leftbottom_grid_id % data->dev_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data->dev_grid_size[1] * data->dev_grid_size[2] + (y + dy) * data->dev_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    get_grid_position(grid_position, grid_id, data);
                    Real weight = grid_particle_quadratic_weight(grid_position, sample_position, data->m_grid_spacing);

                    Real delta_position[3];
                    cudaPhysics::vecSubs3(delta_position, grid_position, sample_position);
                    temp_J += weight * cudaPhysics::dot3(&data->dev_grid_velocity[grid_id * 3], delta_position);
                }
        temp_J = (1 + data->m_time_step * 4 / (data->m_grid_spacing * data->m_grid_spacing) * temp_J) * data->dev_sample_J[i];
    }

    template <typename Real>
    __global__ void P2G_calc_grid_force(StrongPDMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_sample)
            return;
        const Real *sample_position = &data->dev_sample_position[i * 3];
        unsigned int leftbottom_grid_id = data->dev_sample_to_grid_id[i];
        unsigned int x = leftbottom_grid_id / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (leftbottom_grid_id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = leftbottom_grid_id % data->dev_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data->dev_grid_size[1] * data->dev_grid_size[2] + (y + dy) * data->dev_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    get_grid_position(grid_position, grid_id, data);
                    Real weight = grid_particle_quadratic_weight(grid_position, sample_position, data->m_grid_spacing);
                    Real delta_position[3];
                    cudaPhysics::vecSubs3(delta_position, grid_position, sample_position);
                    Real temp = -data->dev_sample_volume[i] 
                            * data->m_lame_lambda 
                            * (data->dev_sample_temp_J[i] - 1)
                            * data->dev_sample_J[i]
                            * 4 / (data->m_grid_spacing * data->m_grid_spacing)
                            * weight;
                    Real force[3];
                    cudaPhysics::vecMul3(force, temp, delta_position);
                    atomicAdd(&data->dev_grid_force[grid_id * 3 + 0], force[0]);
                    atomicAdd(&data->dev_grid_force[grid_id * 3 + 1], force[1]);
                    atomicAdd(&data->dev_grid_force[grid_id * 3 + 2], force[2]);
                }
    }

    template <typename Real>
    __global__ void G2P_calc_sample_force(StrongPDMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_sample)
            return;
        const Real *sample_position = &data->dev_sample_position[i * 3];
        const Real *sample_normal = &data->dev_tri_normal[data->dev_sample_tri_idx[i] * 3];
        Real sample_mass = data->dev_sample_mass[i];
        Real temp_sample_velocity[3] = {0, 0, 0};
        unsigned int leftbottom_grid_id = data->dev_sample_to_grid_id[i];
        unsigned int x = leftbottom_grid_id / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (leftbottom_grid_id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = leftbottom_grid_id % data->dev_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data->dev_grid_size[1] * data->dev_grid_size[2] + (y + dy) * data->dev_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    get_grid_position(grid_position, grid_id, data);
                    Real weight = grid_particle_quadratic_weight(grid_position, sample_position, data->m_grid_spacing);
                    
                    cudaPhysics::axpbypcz(temp_sample_velocity,
                                       (Real)1.0, temp_sample_velocity,
                                       weight, &data->dev_grid_velocity[grid_id * 3],
                                       weight * data->m_time_step / data->dev_grid_mass[grid_id], &data->dev_grid_force[grid_id * 3],
                                       3);
                    // if (i == 77)
                    // {
                    //     printf("grid_id: %u, weight: %.10f, grid_velocity: %.10f %.10f %.10f, grid_force: %.10f %.10f %.10f, force: %.10f %.10f %.10f\n", 
                    //         grid_id, weight, 
                    //         data->dev_grid_velocity[grid_id * 3 + 0], data->dev_grid_velocity[grid_id * 3 + 1], data->dev_grid_velocity[grid_id * 3 + 2],
                    //         data->dev_grid_force[grid_id * 3 + 0], data->dev_grid_force[grid_id * 3 + 1], data->dev_grid_force[grid_id * 3 + 2],
                    //         data->dev_sample_velocity_backup[i * 3 + 0], data->dev_sample_velocity_backup[i * 3 + 1], data->dev_sample_velocity_backup[i * 3 + 2],
                    //         force[0], force[1], force[2]
                    //     );
                    // }
                }
        Real sample_force[3];
        cudaPhysics::vecSubs3(sample_force, temp_sample_velocity, &data->dev_sample_velocity[i * 3]);
        cudaPhysics::vecMul3(sample_force, sample_mass / data->m_time_step);
        Real len = cudaPhysics::dot3(sample_force, sample_normal);
        if (len >= 0)
        {
            len = 0;
        }
        cudaPhysics::vecMul3(&data->dev_sample_force[i * 3], len, sample_normal);
        
        if (i == 77)
        {
            printf("sample_id: %u, force: %.10f %.10f %.10f temp_sample_velocity: %.10f %.10f %.10f sample_velocity: %.10f %.10f %.10f\n", 
                i, 
                data->dev_sample_force[i * 3 + 0], data->dev_sample_force[i * 3 + 1], data->dev_sample_force[i * 3 + 2],
                temp_sample_velocity[0], temp_sample_velocity[1], temp_sample_velocity[2],
                data->dev_sample_velocity[i * 3 + 0], data->dev_sample_velocity[i * 3 + 1], data->dev_sample_velocity[i * 3 + 2]
            );
        }
    }

    template <typename Real>
    __global__ void accumulate_vert_force_from_sample(StrongPDMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_sample)
            return;
        Real *sample_force = &data->dev_sample_force[i * 3];
        for (unsigned int j = 0; j < 3; ++j)
        {
            Real weight = data->dev_sample_barycentric[i * 3 + j];
            unsigned int v = data->dev_triangle[data->dev_sample_tri_idx[i] * 3 + j];
            atomicAdd(&data->dev_vert_force[3 * v + 0], sample_force[0] * weight);
            atomicAdd(&data->dev_vert_force[3 * v + 1], sample_force[1] * weight);
            atomicAdd(&data->dev_vert_force[3 * v + 2], sample_force[2] * weight);
        }
    }

    template <typename Real>
    __global__ void calc_box_collision_force(StrongPDMPMSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            if (data->dev_position[3 * v + j] < data->dev_inner_bbox[j])
            {
                data->dev_vert_force[3 * v + j] += data->m_ground_collision_stiffness * (data->dev_inner_bbox[j] - data->dev_position[3 * v + j]);
                data->dev_constraint_Hessian_diag[v] += data->m_ground_collision_stiffness;
            }
            if (data->dev_position[3 * v + j] > data->dev_inner_bbox[j + 3])
            {
                data->dev_vert_force[3 * v + j] += data->m_ground_collision_stiffness * (data->dev_inner_bbox[j + 3] - data->dev_position[3 * v + j]);
                data->dev_constraint_Hessian_diag[v] += data->m_ground_collision_stiffness;
            }
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
        Real B_const = data->dev_mass[v] * data->m_time_step_inv * data->m_time_step_inv - data->dev_stiffness_matrix_diag[v] + data->dev_constraint_Hessian_diag[v];
        Real inv_B[3];
        inv_B[0] = (Real)1.0 / (B_const + data->dev_collision_Hessian_diag[3 * v + 0]);
        inv_B[1] = (Real)1.0 / (B_const + data->dev_collision_Hessian_diag[3 * v + 1]);
        inv_B[2] = (Real)1.0 / (B_const + data->dev_collision_Hessian_diag[3 * v + 2]);
        Real delta_position[3];
        cudaPhysics::vecMul3(delta_position, inv_B, grad);
        cudaPhysics::vecAdd3(&data->dev_position_next[3 * v], &data->dev_position[3 * v], delta_position);
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
    __global__ void G2P_update_sample_C_J(StrongPDMPMSolverData<Real> *data)
    {
        // unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        // if (i >= data->m_num_sample)
        //     return;
        // Real *sample_position = &data->dev_sample_position[i * 3];
        // unsigned int leftbottom_grid_id = data->dev_sample_to_grid_id[i];
        // unsigned int x = leftbottom_grid_id / data->dev_grid_size[1] / data->dev_grid_size[2];
        // unsigned int y = (leftbottom_grid_id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        // unsigned int z = leftbottom_grid_id % data->dev_grid_size[2];
        // for (unsigned int dx = 0; dx < 3; ++dx)
        //     for (unsigned int dy = 0; dy < 3; ++dy)
        //         for (unsigned int dz = 0; dz < 3; ++dz)
        //         {
        //             unsigned int grid_id = (x + dx) * data->dev_grid_size[1] * data->dev_grid_size[2] + (y + dy) * data->dev_grid_size[2] + (z + dz);
        //             Real grid_position[3];
        //             get_grid_position(grid_position, grid_id, data);
        //             Real weight = grid_particle_quadratic_weight(grid_position, sample_position, data->m_grid_spacing);

        //             Real delta_position[3];
        //             cudaPhysics::vecSubs3(delta_position, grid_position, sample_position);
        //             Real temp_C[9];
        //             cudaPhysics::vecvecT(temp_C, &data->dev_grid_velocity[grid_id * 3], delta_position, 3, 3);
        //             cudaPhysics::matMul3(temp_C, weight * 4 / data->m_grid_spacing / data->m_grid_spacing, temp_C);
        //             cudaPhysics::vecAdd(&data->dev_sample_C[i * 9], temp_C, &data->dev_sample_C[i * 9], 9);
        //             if (i == 77)
        //             {
        //                 printf("grid_id: %u , weight: %f, grid_velocity: [%f, %f, %f], delta_position: [%f, %f, %f]\n", grid_id, weight,
        //                        data->dev_grid_velocity[grid_id * 3 + 0], data->dev_grid_velocity[grid_id * 3 + 1], data->dev_grid_velocity[grid_id * 3 + 2],
        //                        delta_position[0], delta_position[1], delta_position[2]);
        //             }
        //         }

        // data->dev_sample_J[i] = (1 + data->m_time_step * (data->dev_sample_C[i * 9 + 0] + data->dev_sample_C[i * 9 + 4] + data->dev_sample_C[i * 9 + 8])) * data->dev_sample_J[i];
        // if (i == 77)
        // {
        //     printf("sample id: %u, sample_J = %f, sample_C = [%f, %f, %f; %f, %f, %f; %f, %f, %f]\n", i, data->dev_sample_J[i],
        //            data->dev_sample_C[i * 9 + 0], data->dev_sample_C[i * 9 + 1], data->dev_sample_C[i * 9 + 2],
        //            data->dev_sample_C[i * 9 + 3], data->dev_sample_C[i * 9 + 4], data->dev_sample_C[i * 9 + 5],
        //            data->dev_sample_C[i * 9 + 6], data->dev_sample_C[i * 9 + 7], data->dev_sample_C[i * 9 + 8]);
        // }
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
        if (i == 77)
        {
            printf("sample id: %u, sample_J %f\n", i, data->dev_sample_J[i]);
        }
    }

    template <typename Real>
    __global__ void calc_sample_velocity(StrongPDMPMSolverData<Real> *data)
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
        Real *sample_velocity = &data->dev_sample_velocity[i * 3];
        sample_velocity[0] = barycentric[0] * data->dev_velocity[v0 + 0] + barycentric[1] * data->dev_velocity[v1 + 0] + barycentric[2] * data->dev_velocity[v2 + 0];
        sample_velocity[1] = barycentric[0] * data->dev_velocity[v0 + 1] + barycentric[1] * data->dev_velocity[v1 + 1] + barycentric[2] * data->dev_velocity[v2 + 1];
        sample_velocity[2] = barycentric[0] * data->dev_velocity[v0 + 2] + barycentric[1] * data->dev_velocity[v1 + 2] + barycentric[2] * data->dev_velocity[v2 + 2];
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
    const std::vector<Real> &sample_volume,

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
    cudaMalloc((void **)&m_data.dev_collision_Hessian_diag, sizeof(Real) * m_data.m_num_vert * 3);
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
    cudaMalloc((void **)&m_data.dev_tri_normal, sizeof(Real) * m_data.m_num_tri * 3);

    cudaMalloc((void **)&m_data.dev_sample_position, sizeof(Real) * m_data.m_num_sample * 3);
    cudaMalloc((void **)&m_data.dev_sample_barycentric, sizeof(Real) * m_data.m_num_sample * 3);
    cudaMemcpy(m_data.dev_sample_barycentric, sample_barycentric_weights.data(), sizeof(Real) * m_data.m_num_sample * 3, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_sample_tri_idx, sizeof(unsigned int) * m_data.m_num_sample);
    cudaMemcpy(m_data.dev_sample_tri_idx, sample_triangle_idx.data(), sizeof(unsigned int) * m_data.m_num_sample, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_sample_volume, sizeof(Real) * m_data.m_num_sample);
    cudaMemcpy(m_data.dev_sample_volume, sample_volume.data(), sizeof(Real) * m_data.m_num_sample, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_sample_velocity, sizeof(Real) * m_data.m_num_sample * 3);
    cudaMalloc((void **)&m_data.dev_sample_velocity_backup, sizeof(Real) * m_data.m_num_sample * 3);
    cudaMalloc((void **)&m_data.dev_sample_mass, sizeof(Real) * m_data.m_num_sample);
    cudaMalloc((void **)&m_data.dev_sample_J, sizeof(Real) * m_data.m_num_sample);
    cudaMalloc((void **)&m_data.dev_sample_temp_J, sizeof(Real) * m_data.m_num_sample);
    cudaMalloc((void **)&m_data.dev_sample_C, sizeof(Real) * m_data.m_num_sample * 9);
    cudaMemset(m_data.dev_sample_C, 0, sizeof(Real) * m_data.m_num_sample * 9);
    cudaMalloc((void **)&m_data.dev_sample_force, sizeof(Real) * m_data.m_num_sample * 3);
    cudaMalloc((void **)&m_data.dev_sample_diag_K, sizeof(Real) * m_data.m_num_sample * 3);
    cudaMalloc((void **)&m_data.dev_sample_to_grid_id, sizeof(unsigned int) * m_data.m_num_sample);

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
    cudaMalloc(&m_data.dev_grid_force, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_force, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_velocity, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_velocity, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_diag_K, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_diag_K, 0, sizeof(Real) * m_data.m_num_grid * 3);

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
    cudaFree(m_data.dev_collision_Hessian_diag);
    cudaFree(m_data.dev_stiffness_matrix_diag);

    cudaFree(m_data.dev_tetrahedron);
    cudaFree(m_data.dev_tet_density);
    cudaFree(m_data.dev_tet_volume);
    cudaFree(m_data.dev_tet_force);
    cudaFree(m_data.dev_invDm);

    cudaFree(m_data.dev_triangle);
    cudaFree(m_data.dev_tri_normal);

    cudaFree(m_data.dev_sample_position);
    cudaFree(m_data.dev_sample_barycentric);
    cudaFree(m_data.dev_sample_tri_idx);
    cudaFree(m_data.dev_sample_volume);
    cudaFree(m_data.dev_sample_velocity);
    cudaFree(m_data.dev_sample_velocity_backup);
    cudaFree(m_data.dev_sample_mass);
    cudaFree(m_data.dev_sample_J);
    cudaFree(m_data.dev_sample_temp_J);
    cudaFree(m_data.dev_sample_C);
    cudaFree(m_data.dev_sample_force);
    cudaFree(m_data.dev_sample_diag_K);
    cudaFree(m_data.dev_sample_to_grid_id);

    cudaFree(m_data.dev_inner_bbox);
    cudaFree(m_data.dev_outer_bbox);
    cudaFree(m_data.dev_grid_size);
    cudaFree(m_data.dev_grid_momentum);
    cudaFree(m_data.dev_grid_mass);
    cudaFree(m_data.dev_grid_force);
    cudaFree(m_data.dev_grid_velocity);
    cudaFree(m_data.dev_grid_diag_K);

    cudaFree(m_data.dev_gravity);

    cudaFree(m_dev_data);
}

template <typename Real>
void StrongPDMPMSolver<Real>::Step()
{
    StrongPDMPMSolverKernel::calc_triangle_normal<Real><<<CUDA_GRID_SIZE(m_data.m_num_tri), CUDA_BLOCK_SIZE>>>(m_dev_data);
    StrongPDMPMSolverKernel::calc_sample_to_leftbottom_grid_id<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaMemset(m_data.dev_grid_mass, 0, sizeof(Real) * m_data.m_num_grid);
    StrongPDMPMSolverKernel::P2G_mass<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaMemset(m_data.dev_grid_diag_K, 0, sizeof(Real) * m_data.m_num_grid * 3);
    StrongPDMPMSolverKernel::P2G_grid_diag_K<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
    StrongPDMPMSolverKernel::G2P_sample_diag_K<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaMemset(m_data.dev_collision_Hessian_diag, 0, sizeof(Real) * m_data.m_num_vert * 3);
    // StrongPDMPMSolverKernel::accumulate_collision_Hessian_diag_from_sample<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);

    cudaMemcpy(m_data.dev_position_backup, m_data.dev_position, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);
    StrongPDMPMSolverKernel::initial_guess<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaMemcpy(m_data.dev_position_guess, m_data.dev_position, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);
    StrongPDMPMSolverKernel::calc_sample_velocity<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaMemcpy(m_data.dev_sample_velocity_backup, m_data.dev_sample_velocity, sizeof(Real) * m_data.m_num_sample * 3, cudaMemcpyDeviceToDevice);

    Real omega = 1;
    for (unsigned int iter = 0; iter < MAX_ITERATIONS; ++iter)
    {
        // printf("  iteration %u\n", iter);
        // fflush(stdout);

        cudaMemset(m_data.dev_grid_momentum, 0, sizeof(Real) * m_data.m_num_grid * 3);
        cudaMemset(m_data.dev_vert_force, 0, sizeof(Real) * m_data.m_num_vert * 3);
        cudaMemset(m_data.dev_constraint_Hessian_diag, 0, sizeof(Real) * m_data.m_num_vert);
        StrongPDMPMSolverKernel::update_velocity<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
        StrongPDMPMSolverKernel::calc_sample_velocity<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);

        StrongPDMPMSolverKernel::calc_tetrahedron_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
        StrongPDMPMSolverKernel::accumulate_vert_force_from_tet<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);

        StrongPDMPMSolverKernel::P2G_momentum<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
        StrongPDMPMSolverKernel::calc_grids_velocity<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
        StrongPDMPMSolverKernel::G2P_calc_sample_temp_J<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
        StrongPDMPMSolverKernel::P2G_calc_grid_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
        StrongPDMPMSolverKernel::G2P_calc_sample_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
        StrongPDMPMSolverKernel::accumulate_vert_force_from_sample<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);

        StrongPDMPMSolverKernel::calc_box_collision_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);

        StrongPDMPMSolverKernel::jacobi_iteration<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
        UpdateChebyshevOmega(omega, iter);
        StrongPDMPMSolverKernel::Chebyshev<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data, omega);
        SwapPositionBuffers();

        cudaDeviceSynchronize();
    }
    
    StrongPDMPMSolverKernel::update_velocity<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
    StrongPDMPMSolverKernel::calc_sample_velocity<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
    StrongPDMPMSolverKernel::P2G_momentum<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
    StrongPDMPMSolverKernel::calc_grids_velocity<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
    StrongPDMPMSolverKernel::G2P_calc_sample_temp_J<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
    // cudaMemcpy(m_data.dev_sample_J, m_data.dev_sample_temp_J, sizeof(Real) * m_data.m_num_sample, cudaMemcpyDeviceToDevice);
    // StrongPDMPMSolverKernel::G2P_update_sample_C_J<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
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

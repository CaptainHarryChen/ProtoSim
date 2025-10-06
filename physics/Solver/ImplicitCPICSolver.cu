#include "ImplicitCPICSolver.cuh"
#include "CPICTools.cuh"
#include <cuda_utils/cuda_utils.cuh>
#include <Math/algebra.cuh>
#include <Math/geometry.cuh>
#include <Math/elastic_model.cuh>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

namespace ImplicitCPICSolverKernel
{
    template <typename Real>
    __global__ void calc_sample_to_leftbottom_grid_id(ImplicitCPICSolverData<Real> *data)
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
            x = max(0, min(x, (int)data->dev_grid_size[0] - 2));
            y = max(0, min(y, (int)data->dev_grid_size[1] - 2));
            z = max(0, min(z, (int)data->dev_grid_size[2] - 2));
            printf("Warning: sample %d is out of grid, set to %d %d %d [Sample position] %.10f %.10f %.10f\n", i, x, y, z, data->dev_sample_position[i * 3 + 0], data->dev_sample_position[i * 3 + 1], data->dev_sample_position[i * 3 + 2]);
        }

        data->dev_sample_to_grid_id[i] = x * data->dev_grid_size[1] * data->dev_grid_size[2] + y * data->dev_grid_size[2] + z;
    }

    template <typename Real>
    __global__ void sample_to_grid(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_sample)
            return;
        unsigned int tri_idx = data->dev_sample_tri_idx[i];
        const unsigned int *tri = &data->dev_triangle[tri_idx * 3];
        const Real *tri_a_pos = &data->dev_position[tri[0] * 3];
        const Real *tri_b_pos = &data->dev_position[tri[1] * 3];
        const Real *tri_c_pos = &data->dev_position[tri[2] * 3];
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
                    CPICTools::get_grid_position(grid_position, grid_id, data->dev_grid_size, data->dev_outer_bbox, data->m_grid_spacing);
                    if (cudaPhysics::is_point_in_triangle(grid_position, tri_a_pos, tri_b_pos, tri_c_pos))
                    {
                        float distance = cudaPhysics::point_to_triangle_sign_distance(grid_position, tri_a_pos, tri_b_pos, tri_c_pos);
                        bool inside = distance < 0;
                        distance = inside ? -distance : distance;
                        uint64_t packed_info = CPICTools::pack_tri_info(distance, inside, tri_idx);
                        atomicMin(&data->dev_grid_tri_info[grid_id], packed_info);
                    }
                }
    }

    template <typename Real>
    __global__ void update_F(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        if (data->dev_particle_type[i] == MPM_ELASTIC)
        {
            Real F[9], temp[9];
            cudaPhysics::matMul3(temp, data->m_time_step, &data->dev_particle_C[i * 9]);
            temp[0] += 1.;
            temp[4] += 1.;
            temp[8] += 1.;
            cudaPhysics::matMul3(F, temp, &data->dev_particle_F[i * 9]);
            cudaPhysics::vecCopy(&data->dev_particle_F[i * 9], F, 9);
        }
        else if (data->dev_particle_type[i] == MPM_FLUID)
        {
            // Only use the first element of F to store J
            Real J = data->dev_particle_F[i * 9] * (1 + data->m_time_step * (data->dev_particle_C[i * 9] + data->dev_particle_C[i * 9 + 4] + data->dev_particle_C[i * 9 + 8]));
            data->dev_particle_F[i * 9 + 0] = J;
        }
    }

    template <typename Real>
    __global__ void calc_particle_to_leftbottom_grid_id(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        // find the nearest grid, so plus 0.5 and floor
        int x = floor((data->dev_particle_position[i * 3 + 0] - data->dev_outer_bbox[0]) / data->m_grid_spacing + 0.5);
        int y = floor((data->dev_particle_position[i * 3 + 1] - data->dev_outer_bbox[1]) / data->m_grid_spacing + 0.5);
        int z = floor((data->dev_particle_position[i * 3 + 2] - data->dev_outer_bbox[2]) / data->m_grid_spacing + 0.5);
        // get the left bottom grid id
        x--;
        y--;
        z--;
        if (x < 0 || y < 0 || z < 0 || x >= data->dev_grid_size[0] - 2 || y >= data->dev_grid_size[1] - 2 || z >= data->dev_grid_size[2] - 2)
        {
            x = max(0, min(x, (int)data->dev_grid_size[0] - 2));
            y = max(0, min(y, (int)data->dev_grid_size[1] - 2));
            z = max(0, min(z, (int)data->dev_grid_size[2] - 2));
            printf("Warning: particle %d is out of grid, set to %d %d %d [Particle position] %.10f %.10f %.10f\n", i, x, y, z, data->dev_particle_position[i * 3 + 0], data->dev_particle_position[i * 3 + 1], data->dev_particle_position[i * 3 + 2]);
        }

        data->dev_particle_to_grid_id[i] = x * data->dev_grid_size[1] * data->dev_grid_size[2] + y * data->dev_grid_size[2] + z;
    }

    template <typename Real>
    __global__ void P2G_momentum_and_mass_and_K(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        Real *particle_position = &data->dev_particle_position[i * 3];
        Real *particle_velocity = &data->dev_particle_velocity[i * 3];
        unsigned int leftbottom_grid_id = data->dev_particle_to_grid_id[i];
        unsigned int x = leftbottom_grid_id / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (leftbottom_grid_id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = leftbottom_grid_id % data->dev_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data->dev_grid_size[1] * data->dev_grid_size[2] + (y + dy) * data->dev_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    CPICTools::get_grid_position(grid_position, grid_id, data->dev_grid_size, data->dev_outer_bbox, data->m_grid_spacing);
                    Real weight = CPICTools::grid_particle_quadratic_weight(grid_position, particle_position, data->m_grid_spacing);
                    Real momentum[3];
                    Real delta_position[3];
                    cudaPhysics::vecSubs3(delta_position, grid_position, particle_position);
                    cudaPhysics::matVec3(momentum, &data->dev_particle_C[i * 9], delta_position);
                    cudaPhysics::vecMul3(momentum, data->dev_particle_mass[i], momentum);

                    Real temp_momentum[3];
                    cudaPhysics::vecMul3(temp_momentum, data->dev_particle_mass[i], particle_velocity);
                    cudaPhysics::vecAdd3(momentum, temp_momentum, momentum);
                    cudaPhysics::vecMul3(momentum, weight, momentum);
                    atomicAdd(&data->dev_grid_momentum[grid_id * 3 + 0], momentum[0]);
                    atomicAdd(&data->dev_grid_momentum[grid_id * 3 + 1], momentum[1]);
                    atomicAdd(&data->dev_grid_momentum[grid_id * 3 + 2], momentum[2]);

                    atomicAdd(&data->dev_grid_mass[grid_id], weight * data->dev_particle_mass[i]);
                    
                    Real diag_K[3];
                    cudaPhysics::vecMul3(diag_K, delta_position, delta_position);
                    cudaPhysics::vecMul3(diag_K, -data->dev_particle_volume[i]
                                                * data->m_lame_lambda
                                                * data->dev_particle_F[i * 9] * data->dev_particle_F[i * 9]
                                                * 16 * weight * weight
                                                / (data->m_grid_spacing * data->m_grid_spacing * data->m_grid_spacing * data->m_grid_spacing));
                    atomicAdd(&data->dev_grid_diag_Kv[grid_id * 3 + 0], diag_K[0]);
                    atomicAdd(&data->dev_grid_diag_Kv[grid_id * 3 + 1], diag_K[1]);
                    atomicAdd(&data->dev_grid_diag_Kv[grid_id * 3 + 2], diag_K[2]);
                }
    }

    template <typename Real>
    __global__ void calc_grids_velocity(ImplicitCPICSolverData<Real> *data)
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
    __global__ void grids_gravity(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        Real delta_velocity[3];
        cudaPhysics::vecMul3(delta_velocity, data->m_time_step, data->dev_gravity);
        cudaPhysics::vecAdd3(&data->dev_grid_velocity[i * 3], &data->dev_grid_velocity[i * 3], delta_velocity);
    }

    template <typename Real>
    __global__ void particles_gravity(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        Real delta_velocity[3];
        cudaPhysics::vecMul3(delta_velocity, data->m_time_step, data->dev_gravity);
        cudaPhysics::vecAdd3(&data->dev_particle_velocity[i * 3], &data->dev_particle_velocity[i * 3], delta_velocity);
    }

    template <typename Real>
    __global__ void G2P_calc_particle_temp_F(ImplicitCPICSolverData<Real> *data, Real alpha = 0)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        assert(data->dev_particle_type[i] == MPM_FLUID);
        Real &temp_J = data->dev_particle_temp_J[i];
        temp_J = 0;
        Real *particle_position = &data->dev_particle_position[i * 3];
        unsigned int leftbottom_grid_id = data->dev_particle_to_grid_id[i];
        unsigned int x = leftbottom_grid_id / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (leftbottom_grid_id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = leftbottom_grid_id % data->dev_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data->dev_grid_size[1] * data->dev_grid_size[2] + (y + dy) * data->dev_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    CPICTools::get_grid_position(grid_position, grid_id, data->dev_grid_size, data->dev_outer_bbox, data->m_grid_spacing);
                    Real weight = CPICTools::grid_particle_quadratic_weight(grid_position, particle_position, data->m_grid_spacing);

                    Real delta_position[3], vel[3];
                    cudaPhysics::vecSubs3(delta_position, grid_position, particle_position);
                    cudaPhysics::axpby(vel, (Real)1, &data->dev_grid_velocity[grid_id * 3], alpha, &data->dev_grid_velocity_delta[grid_id * 3], 3);
                    temp_J += weight * cudaPhysics::dot3(vel, delta_position);
                }
        temp_J = (1 + data->m_time_step * 4 / (data->m_grid_spacing * data->m_grid_spacing) * temp_J) * data->dev_particle_F[i * 9];
    }

    template <typename Real>
    __global__ void P2G_calc_grid_force(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        Real *particle_position = &data->dev_particle_position[i * 3];
        Real *particle_velocity = &data->dev_particle_velocity[i * 3];
        unsigned int leftbottom_grid_id = data->dev_particle_to_grid_id[i];
        unsigned int x = leftbottom_grid_id / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (leftbottom_grid_id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = leftbottom_grid_id % data->dev_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data->dev_grid_size[1] * data->dev_grid_size[2] + (y + dy) * data->dev_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    CPICTools::get_grid_position(grid_position, grid_id, data->dev_grid_size, data->dev_outer_bbox, data->m_grid_spacing);
                    Real weight = CPICTools::grid_particle_quadratic_weight(grid_position, particle_position, data->m_grid_spacing);
                    Real delta_position[3];
                    cudaPhysics::vecSubs3(delta_position, grid_position, particle_position);
                    Real temp = -data->dev_particle_volume[i] 
                            * data->m_lame_lambda 
                            * (data->dev_particle_temp_J[i] - 1)
                            * data->dev_particle_F[i * 9]
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
    __global__ void grid_op(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        Real *diag_B = &data->dev_grid_diag_B[i * 3];
        Real *grid_force = &data->dev_grid_force[i * 3];
        diag_B[0] = diag_B[1] = diag_B[2] = (Real)1;
        if (data->dev_grid_mass[i] <= 0)
            return;
        cudaPhysics::axpby(diag_B, (Real)1, diag_B, -data->m_time_step / data->dev_grid_mass[i], &data->dev_grid_diag_Kv[i * 3], 3);

        unsigned int x = i / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (i / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = i % data->dev_grid_size[2];
        Real grid_pos[3];
        CPICTools::get_grid_position(grid_pos, i, data->dev_grid_size, data->dev_outer_bbox, data->m_grid_spacing);
        cudaPhysics::axpby(grid_pos, (Real)1, grid_pos, data->m_time_step, &data->dev_grid_velocity[i * 3], 3);
        for (unsigned int j = 0; j < 3; j++)
        {
            if (grid_pos[j] <= data->dev_inner_bbox[j])
            {
                diag_B[j] += data->m_grid_box_collision_stiffness * data->m_time_step * data->m_time_step;
                grid_force[j] += data->dev_grid_mass[i] * data->m_grid_box_collision_stiffness * (data->dev_inner_bbox[j] - grid_pos[j]);
            }
            if (grid_pos[j] >= data->dev_inner_bbox[j + 3])
            {
                diag_B[j] += data->m_grid_box_collision_stiffness * data->m_time_step * data->m_time_step;
                grid_force[j] += data->dev_grid_mass[i] * data->m_grid_box_collision_stiffness * (data->dev_inner_bbox[j + 3] - grid_pos[j]);
            }
        }
    }

    template <typename Real>
    __global__ void inside_grids_force(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid || data->dev_grid_tri_info[i] == UINT64_MAX || data->dev_grid_mass[i] <= 0)
            return;
        float _;
        bool __;
        unsigned int tri_idx;
        CPICTools::unpack_tri_info(data->dev_grid_tri_info[i], _, __, tri_idx);

        const unsigned int *tri = &data->dev_triangle[tri_idx * 3];
        // dev_position is x^(k), which has already been updated by velocity
        const Real *tri_a_pos = &data->dev_position[tri[0] * 3];
        const Real *tri_b_pos = &data->dev_position[tri[1] * 3];
        const Real *tri_c_pos = &data->dev_position[tri[2] * 3];

        Real grid_position[3];
        CPICTools::get_grid_position(grid_position, i, data->dev_grid_size, data->dev_outer_bbox, data->m_grid_spacing);
        // grid_position need to be updated by velocity in implicit euler
        cudaPhysics::axpby(grid_position, (Real)1, grid_position, data->m_time_step, &data->dev_grid_velocity[i * 3], 3);
        
        Real cur_dis = cudaPhysics::point_to_triangle_sign_distance(grid_position, tri_a_pos, tri_b_pos, tri_c_pos);
        if (cur_dis < 0)
        {
            Real bary[3], normal[3], force[3], hessian[3];
            cudaPhysics::triangle_normal(normal, tri_a_pos, tri_b_pos, tri_c_pos);
            cudaPhysics::projection_barycentric_coordinates(bary, grid_position, tri_a_pos, tri_b_pos, tri_c_pos);
            cudaPhysics::vecMul3(force, data->m_couple_collision_stiffness * data->dev_grid_mass[i] * (-cur_dis), normal);
            cudaPhysics::vecMul3(hessian, normal, normal);
            cudaPhysics::vecMul3(hessian, data->m_couple_collision_stiffness * data->dev_grid_mass[i], hessian);

            cudaPhysics::axpby(&data->dev_grid_diag_B[i * 3], (Real)1, &data->dev_grid_diag_B[i * 3], (Real)1 / data->dev_grid_mass[i] * data->m_time_step * data->m_time_step, hessian, 3);
            cudaPhysics::vecAdd3(&data->dev_grid_force[i * 3], &data->dev_grid_force[i * 3], force);

            for (unsigned int j = 0; j < 3; ++j)
                for (unsigned int k = 0; k < 3; ++k)
                {
                    atomicAdd(&data->dev_vert_force[tri[j] * 3 + k], -bary[j] * force[k]);
                    atomicAdd(&data->dev_constraint_diag_K[tri[j] * 3 + k], bary[j] * hessian[k]);
                }
        }
    }

    template <typename Real>
    __global__ void mpm_jacobi_iteration(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        Real grad[3];
        Real tmp = data->dev_grid_mass[i] > 0 ? data->m_time_step / data->dev_grid_mass[i] : 0;
        cudaPhysics::axpbypcz(grad, (Real)1, &data->dev_grid_velocity_hat[i * 3],
                                    (Real)-1, &data->dev_grid_velocity[i * 3],
                                    tmp, &data->dev_grid_force[i * 3],
                            3);
        // atomicAdd(data->dev_residual, (grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]) / data->m_num_grid);
        Real inv_B[3];
        inv_B[0] = (Real)1.0 / data->dev_grid_diag_B[i * 3 + 0];
        inv_B[1] = (Real)1.0 / data->dev_grid_diag_B[i * 3 + 1];
        inv_B[2] = (Real)1.0 / data->dev_grid_diag_B[i * 3 + 2];
        cudaPhysics::vecMul3(&data->dev_grid_velocity_delta[3 * i], inv_B, grad);
    }

    template <typename Real>
    __global__ void MPM_Chebyshev(ImplicitCPICSolverData<Real> *data, Real omega)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        Real delta_velocity[3];
        cudaPhysics::vecSubs3(delta_velocity, &data->dev_grid_velocity_next[3 * i], &data->dev_grid_velocity[3 * i]);
        cudaPhysics::axpby(&data->dev_grid_velocity_next[3 * i], data->m_under_relaxation, delta_velocity, (Real)1.0, &data->dev_grid_velocity[3 * i], 3);
        cudaPhysics::vecSubs3(delta_velocity, &data->dev_grid_velocity_next[3 * i], &data->dev_grid_velocity_prev[3 * i]);
        cudaPhysics::axpby(&data->dev_grid_velocity_next[3 * i], omega, delta_velocity, (Real)1.0, &data->dev_grid_velocity_prev[3 * i], 3);
    }

    template <typename Real>
    __global__ void G2P_velocity_and_C(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        if (data->dev_particle_type[i] == MPM_STATIC)
            return;
        const Real *particle_position = &data->dev_particle_position[i * 3];
        unsigned int leftbottom_grid_id = data->dev_particle_to_grid_id[i];
        unsigned int x = leftbottom_grid_id / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (leftbottom_grid_id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = leftbottom_grid_id % data->dev_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data->dev_grid_size[1] * data->dev_grid_size[2] + (y + dy) * data->dev_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    CPICTools::get_grid_position(grid_position, grid_id, data->dev_grid_size, data->dev_outer_bbox, data->m_grid_spacing);
                    Real weight = CPICTools::grid_particle_quadratic_weight(grid_position, particle_position, data->m_grid_spacing);

                    Real velocity[3];
                    cudaPhysics::vecMul3(velocity, weight, &data->dev_grid_velocity[grid_id * 3]);
                    cudaPhysics::vecAdd3(&data->dev_particle_velocity[i * 3], &data->dev_particle_velocity[i * 3], velocity);

                    Real delta_position[3];
                    cudaPhysics::vecSubs3(delta_position, grid_position, particle_position);
                    Real temp_C[9];
                    cudaPhysics::vecvecT(temp_C, &data->dev_grid_velocity[grid_id * 3], delta_position, 3, 3);
                    cudaPhysics::matMul3(temp_C, weight * 4 / data->m_grid_spacing / data->m_grid_spacing, temp_C);
                    cudaPhysics::vecAdd(&data->dev_particle_C[i * 9], temp_C, &data->dev_particle_C[i * 9], 9);
                }
    }

    template <typename Real>
    __global__ void update_particle_positions(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        if (data->dev_particle_type[i] == 0)
            return;
        Real *position = &data->dev_particle_position[i * 3];
        Real delta_position[3];
        cudaPhysics::vecMul3(delta_position, data->m_time_step, &data->dev_particle_velocity[i * 3]);
        cudaPhysics::vecAdd3(position, position, delta_position);
    }

    template <typename Real>
    __global__ void grids_boundary_conditions(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        unsigned int x = i / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (i / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = i % data->dev_grid_size[2];
        Real grid_pos[3];
        CPICTools::get_grid_position(grid_pos, i, data->dev_grid_size, data->dev_outer_bbox, data->m_grid_spacing);
        for (unsigned int j = 0; j < 3; j++)
        {
            if (grid_pos[j] + data->dev_grid_velocity[i * 3 + j] * data->m_time_step < data->dev_inner_bbox[j])
            {
                data->dev_grid_velocity[i * 3 + j] = (data->dev_inner_bbox[j] - grid_pos[j]) * data->m_time_step_inv;
            }
            if (grid_pos[j] + data->dev_grid_velocity[i * 3 + j] * data->m_time_step > data->dev_inner_bbox[j + 3])
            {
                data->dev_grid_velocity[i * 3 + j] = (data->dev_inner_bbox[j + 3] - grid_pos[j]) * data->m_time_step_inv;
            }
        }
    }

    template <typename Real>
    __global__ void particles_boundary_conditions(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            if (data->dev_particle_position[i * 3 + j] < data->dev_inner_bbox[j])
            {
                data->dev_particle_position[i * 3 + j] = data->dev_inner_bbox[j];
            }
            if (data->dev_particle_position[i * 3 + j] > data->dev_inner_bbox[j + 3])
            {
                data->dev_particle_position[i * 3 + j] = data->dev_inner_bbox[j + 3];
            }
        }
    }

    template <typename Real>
    __global__ void tetrahedron_initialize(ImplicitCPICSolverData<Real> *data)
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
    __global__ void tet_stiffness_matrix_diag(ImplicitCPICSolverData<Real> *data)
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
    __global__ void initial_guess(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        Real *force = data->dev_gravity;
        cudaPhysics::axpby(&data->dev_velocity[3 * v], (Real)1.0, &data->dev_velocity[3 * v], data->m_time_step, force, 3);
    }

    template <typename Real>
    __global__ void calc_tetrahedron_force(ImplicitCPICSolverData<Real> *data)
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
    __global__ void accumulate_vert_force(ImplicitCPICSolverData<Real> *data)
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
    __global__ void calc_box_collision_force(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            if (data->dev_position[3 * v + j] < data->dev_inner_bbox[j])
            {
                data->dev_vert_force[3 * v + j] += data->m_solid_box_collision_stiffness * (data->dev_inner_bbox[j] - data->dev_position[3 * v + j]);
                data->dev_constraint_diag_K[3 * v + j] += data->m_solid_box_collision_stiffness;
            }
            if (data->dev_position[3 * v + j] > data->dev_inner_bbox[j + 3])
            {
                data->dev_vert_force[3 * v + j] += data->m_solid_box_collision_stiffness * (data->dev_inner_bbox[j + 3] - data->dev_position[3 * v + j]);
                data->dev_constraint_diag_K[3 * v + j] += data->m_solid_box_collision_stiffness;
            }
        }
    }

    template <typename Real>
    __global__ void pd_jacobi_iteration(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        Real grad[3];
        cudaPhysics::axpbypcz(grad, (Real)1, &data->dev_velocity_hat[3 * v], (Real)-1, &data->dev_velocity[3 * v],
                                    data->m_time_step / data->dev_mass[v], &data->dev_vert_force[3 * v], 3);
        // atomicAdd(data->dev_residual, (grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]) / data->m_num_vert);
        Real B[3];
        B[0] = B[1] = B[2] = 1 - data->m_time_step * data->m_time_step / data->dev_mass[v] * data->dev_stiffness_matrix_diag[v];
        cudaPhysics::axpby(B, (Real)1, B, data->m_time_step * data->m_time_step / data->dev_mass[v], &data->dev_constraint_diag_K[3 * v], 3);
        Real invB[3];
        invB[0] = (Real)1.0 / B[0];
        invB[1] = (Real)1.0 / B[1];
        invB[2] = (Real)1.0 / B[2];
        cudaPhysics::vecMul3(&data->dev_velocity_delta[3 * v], invB, grad);
    }

    template <typename Real>
    __global__ void calc_particle_internal_energy(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        // fluid potential energy
        atomicAdd(data->dev_energy, 0.5 * data->m_lame_lambda * (data->dev_particle_temp_J[i] - 1) * (data->dev_particle_temp_J[i] - 1) * data->dev_particle_volume[i]);
    }

    template <typename Real>
    __global__ void calc_grid_energy(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        Real energy = 0;
        Real delta_velocity[3];
        cudaPhysics::vecSubs3(delta_velocity, &data->dev_grid_velocity_next[i * 3], &data->dev_grid_velocity_hat[i * 3]);
        energy += 0.5 * data->dev_grid_mass[i] * data->m_time_step_inv * cudaPhysics::dot3(delta_velocity, delta_velocity); // inertial energy
        
        // box constraint potential energy
        unsigned int x = i / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (i / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = i % data->dev_grid_size[2];
        Real grid_pos[3];
        CPICTools::get_grid_position(grid_pos, i, data->dev_grid_size, data->dev_outer_bbox, data->m_grid_spacing);
        cudaPhysics::axpby(grid_pos, (Real)1, grid_pos, data->m_time_step, &data->dev_grid_velocity_next[i * 3], 3);
        for (unsigned int j = 0; j < 3; j++)
        {
            if (grid_pos[j] <= data->dev_inner_bbox[j])
            {
                energy += 0.5 * data->dev_grid_mass[i] * data->m_grid_box_collision_stiffness * (data->dev_inner_bbox[j] - grid_pos[j]) * (data->dev_inner_bbox[j] - grid_pos[j]);
            }
            if (grid_pos[j] >= data->dev_inner_bbox[j + 3])
            {
                energy += 0.5 * data->dev_grid_mass[i] * data->m_grid_box_collision_stiffness * (data->dev_inner_bbox[j + 3] - grid_pos[j]) * (data->dev_inner_bbox[j + 3] - grid_pos[j]);
            }
        }
        atomicAdd(data->dev_energy, energy);
    }

    template <typename Real>
    __global__ void calc_tet_energy(ImplicitCPICSolverData<Real> *data)
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
        atomicAdd(data->dev_energy, data->dev_tet_volume[t] * data->m_lame_mu * cudaPhysics::dot(FSubR, FSubR, 9));
    }

    template <typename Real>
    __global__ void calc_vert_energy(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        Real delta_v[3];
        cudaPhysics::vecSubs3(delta_v, &data->dev_velocity_next[3 * v], &data->dev_velocity_hat[3 * v]);
        Real energy = 0.5 * data->dev_mass[v] * data->m_time_step_inv * cudaPhysics::dot3(delta_v, delta_v); // inertial energy

        // box constraint potential energy
        for (unsigned int j = 0; j < 3; ++j)
        {
            if (data->dev_position[3 * v + j] < data->dev_inner_bbox[j])
            {
                energy += 0.5 * data->m_solid_box_collision_stiffness * (data->dev_inner_bbox[j] - data->dev_position[3 * v + j]) * (data->dev_inner_bbox[j] - data->dev_position[3 * v + j]);
            }
            if (data->dev_position[3 * v + j] > data->dev_inner_bbox[j + 3])
            {
                energy += 0.5 * data->m_solid_box_collision_stiffness * (data->dev_inner_bbox[j + 3] - data->dev_position[3 * v + j]) * (data->dev_inner_bbox[j + 3] - data->dev_position[3 * v + j]);
            }
        }
    }

    template <typename Real>
    __global__ void calc_couple_energy(ImplicitCPICSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid || data->dev_grid_tri_info[i] == UINT64_MAX || data->dev_grid_mass[i] <= 0)
            return;
        float _;
        bool __;
        unsigned int tri_idx;
        CPICTools::unpack_tri_info(data->dev_grid_tri_info[i], _, __, tri_idx);

        const unsigned int *tri = &data->dev_triangle[tri_idx * 3];
        // dev_position is x^(k), which has already been updated by velocity
        const Real *tri_a_pos = &data->dev_position[tri[0] * 3];
        const Real *tri_b_pos = &data->dev_position[tri[1] * 3];
        const Real *tri_c_pos = &data->dev_position[tri[2] * 3];

        Real grid_position[3];
        CPICTools::get_grid_position(grid_position, i, data->dev_grid_size, data->dev_outer_bbox, data->m_grid_spacing);
        // grid_position need to be updated by velocity in implicit euler
        cudaPhysics::axpby(grid_position, (Real)1, grid_position, data->m_time_step, &data->dev_grid_velocity[i * 3], 3);
        
        Real cur_dis = cudaPhysics::point_to_triangle_sign_distance(grid_position, tri_a_pos, tri_b_pos, tri_c_pos);
        if (cur_dis < 0)
        {
            Real bary[3], normal[3], force[3], hessian[3];
            cudaPhysics::triangle_normal(normal, tri_a_pos, tri_b_pos, tri_c_pos);
            cudaPhysics::projection_barycentric_coordinates(bary, grid_position, tri_a_pos, tri_b_pos, tri_c_pos);
            atomicAdd(data->dev_energy, 0.5 * data->m_couple_collision_stiffness * data->dev_grid_mass[i] * cur_dis * cur_dis); // potential energy
        }
    }

    template <typename Real>
    __global__ void PD_Chebyshev(ImplicitCPICSolverData<Real> *data, Real omega)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        Real delta_velocity[3];
        cudaPhysics::vecSubs3(delta_velocity, &data->dev_velocity_next[3 * v], &data->dev_velocity[3 * v]);
        cudaPhysics::axpby(&data->dev_velocity_next[3 * v], data->m_under_relaxation, delta_velocity, (Real)1.0, &data->dev_velocity[3 * v], 3);
        cudaPhysics::vecSubs3(delta_velocity, &data->dev_velocity_next[3 * v], &data->dev_velocity_prev[3 * v]);
        cudaPhysics::axpby(&data->dev_velocity_next[3 * v], omega, delta_velocity, (Real)1.0, &data->dev_velocity_prev[3 * v], 3);
    }

    template <typename Real>
    __global__ void calc_sample_position(ImplicitCPICSolverData<Real> *data)
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
ImplicitCPICSolver<Real>::ImplicitCPICSolver(
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
)
{
    m_data.m_num_vert = (unsigned int)node_position.size() / 3;
    m_data.m_num_tet = (unsigned int)tetrahedron.size() / 4;
    m_data.m_num_tri = (unsigned int)surface_triangle.size() / 3;
    m_data.m_num_sample = (unsigned int)sample_barycentric_weights.size() / 3;
    assert(m_data.m_num_sample == sample_triangle_idx.size());
    printf("num_vert = %u, num_tet = %u, num_tri = %u, num_sample = %u\n", m_data.m_num_vert, m_data.m_num_tet, m_data.m_num_tri, m_data.m_num_sample);

    cudaMalloc((void **)&m_data.dev_position_backup, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_position, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemcpy(m_data.dev_position, node_position.data(), sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_velocity_hat, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_velocity_prev, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_velocity, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemset(m_data.dev_velocity, 0, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_velocity_next, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_velocity_delta, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_mass, sizeof(Real) * m_data.m_num_vert);
    cudaMemset(m_data.dev_mass, 0, sizeof(Real) * m_data.m_num_vert);
    cudaMalloc((void **)&m_data.dev_vert_force, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_constraint_diag_K, sizeof(Real) * m_data.m_num_vert * 3);
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

    cudaMalloc((void **)&m_data.dev_triangle, sizeof(unsigned int) * surface_triangle.size());
    cudaMemcpy(m_data.dev_triangle, surface_triangle.data(), sizeof(unsigned int) * surface_triangle.size(), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&m_data.dev_sample_position, sizeof(Real) * m_data.m_num_sample * 3);
    cudaMalloc((void **)&m_data.dev_sample_barycentric, sizeof(Real) * sample_barycentric_weights.size());
    cudaMemcpy(m_data.dev_sample_barycentric, sample_barycentric_weights.data(), sizeof(Real) * sample_barycentric_weights.size(), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_sample_tri_idx, sizeof(unsigned int) * sample_triangle_idx.size());
    cudaMemcpy(m_data.dev_sample_tri_idx, sample_triangle_idx.data(), sizeof(unsigned int) * sample_triangle_idx.size(), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_sample_to_grid_id, sizeof(unsigned int) * m_data.m_num_sample);

    assert(particle_position.size() % 3 == 0);
    m_data.m_num_particle = (unsigned int)particle_position.size() / 3;
    printf("num_particle = %u\n", m_data.m_num_particle);

    cudaMalloc(&m_data.dev_particle_position, sizeof(Real) * m_data.m_num_particle * 3);
    cudaMemcpy(m_data.dev_particle_position, particle_position.data(), sizeof(Real) * m_data.m_num_particle * 3, cudaMemcpyHostToDevice);
    cudaMalloc(&m_data.dev_particle_velocity, sizeof(Real) * m_data.m_num_particle * 3);
    cudaMemset(m_data.dev_particle_velocity, 0, sizeof(Real) * m_data.m_num_particle * 3);
    cudaMalloc(&m_data.dev_particle_mass, sizeof(Real) * particle_mass.size());
    cudaMemcpy(m_data.dev_particle_mass, particle_mass.data(), sizeof(Real) * m_data.m_num_particle, cudaMemcpyHostToDevice);
    cudaMalloc(&m_data.dev_particle_volume, sizeof(Real) * particle_volume.size());
    cudaMemcpy(m_data.dev_particle_volume, particle_volume.data(), sizeof(Real) * m_data.m_num_particle, cudaMemcpyHostToDevice);
    cudaMalloc(&m_data.dev_particle_type, sizeof(unsigned int) * particle_type.size());
    cudaMemcpy(m_data.dev_particle_type, particle_type.data(), sizeof(unsigned int) * m_data.m_num_particle, cudaMemcpyHostToDevice);
    cudaMalloc(&m_data.dev_particle_C, sizeof(Real) * m_data.m_num_particle * 9);
    cudaMemset(m_data.dev_particle_C, 0, sizeof(Real) * m_data.m_num_particle * 9);
    cudaMalloc(&m_data.dev_particle_temp_J, sizeof(Real) * m_data.m_num_particle);
    cudaMalloc(&m_data.dev_particle_F, sizeof(Real) * m_data.m_num_particle * 9);
    cudaPhysics::fill_identity_matrix(m_data.dev_particle_F, m_data.m_num_particle, 3);
    cudaMalloc(&m_data.dev_particle_to_grid_id, sizeof(unsigned int) * m_data.m_num_particle);

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
    printf("grid_size = [%u %u %u]\n", grid_size[0], grid_size[1], grid_size[2]);
    for (unsigned int i = 0; i < 3; ++i)
        outer_bbox[i + 3] = outer_bbox[i] + m_data.m_grid_spacing * grid_size[i];
    m_data.m_num_grid = grid_size[0] * grid_size[1] * grid_size[2];
    printf("num_grid = %u\n", m_data.m_num_grid);

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
    cudaMalloc(&m_data.dev_grid_velocity, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_velocity, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_tri_info, sizeof(uint64_t) * m_data.m_num_grid);
    cudaMemset(m_data.dev_grid_tri_info, 0xFF, sizeof(uint64_t) * m_data.m_num_grid);
    cudaMalloc(&m_data.dev_grid_velocity_hat, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_velocity_prev, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_velocity_delta, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_velocity_next, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_diag_Kv, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_diag_B, sizeof(Real) * m_data.m_num_grid * 3);

    cudaMalloc(&m_data.dev_energy, sizeof(Real));
    cudaMalloc(&m_data.dev_residual, sizeof(Real));

    m_data.m_time_step = TIME_STEP;
    m_data.m_time_step_inv = 1.0f / m_data.m_time_step;
    m_data.m_lame_mu = LAME_MU;
    m_data.m_lame_lambda = LAME_LAMBDA;
    m_data.m_grid_box_collision_stiffness = GRID_BOX_COLLISION_STIFFNESS;
    m_data.m_solid_box_collision_stiffness = SOLID_BOX_COLLISION_STIFFNESS;
    m_data.m_couple_collision_stiffness = COUPLE_COLLISION_STIFFNESS;
    m_data.m_under_relaxation = UNDER_RELAXATION;
    cudaMalloc(&m_data.dev_gravity, sizeof(Real) * 3);
    std::vector<Real> gravity = {0., -GRAVITY, 0.};
    cudaMemcpy(m_data.dev_gravity, gravity.data(), sizeof(Real) * 3, cudaMemcpyHostToDevice);

    cudaMalloc(&m_dev_data, sizeof(ImplicitCPICSolverData<Real>));
    cudaMemcpy(m_dev_data, &m_data, sizeof(ImplicitCPICSolverData<Real>), cudaMemcpyHostToDevice);

    ImplicitCPICSolverKernel::tetrahedron_initialize<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
    ImplicitCPICSolverKernel::tet_stiffness_matrix_diag<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
    ImplicitCPICSolverKernel::calc_sample_position<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
}

template <typename Real>
ImplicitCPICSolver<Real>::~ImplicitCPICSolver()
{
    cudaFree(m_data.dev_particle_position);
    cudaFree(m_data.dev_particle_velocity);
    cudaFree(m_data.dev_particle_mass);
    cudaFree(m_data.dev_particle_volume);
    cudaFree(m_data.dev_particle_type);
    cudaFree(m_data.dev_particle_C);
    cudaFree(m_data.dev_particle_temp_J);
    cudaFree(m_data.dev_particle_F);
    cudaFree(m_data.dev_particle_to_grid_id);
    cudaFree(m_data.dev_inner_bbox);
    cudaFree(m_data.dev_outer_bbox);
    cudaFree(m_data.dev_grid_size);
    cudaFree(m_data.dev_grid_momentum);
    cudaFree(m_data.dev_grid_mass);
    cudaFree(m_data.dev_grid_velocity);
    cudaFree(m_data.dev_grid_force);
    cudaFree(m_data.dev_grid_tri_info);
    cudaFree(m_data.dev_grid_velocity_hat);
    cudaFree(m_data.dev_grid_velocity_prev);
    cudaFree(m_data.dev_grid_velocity_delta);
    cudaFree(m_data.dev_grid_velocity_next);
    cudaFree(m_data.dev_grid_diag_Kv);
    cudaFree(m_data.dev_grid_diag_B);

    cudaFree(m_data.dev_position_backup);
    cudaFree(m_data.dev_position);
    cudaFree(m_data.dev_velocity_hat);
    cudaFree(m_data.dev_velocity_prev);
    cudaFree(m_data.dev_velocity);
    cudaFree(m_data.dev_velocity_next);
    cudaFree(m_data.dev_velocity_delta);
    cudaFree(m_data.dev_mass);
    cudaFree(m_data.dev_vert_force);
    cudaFree(m_data.dev_constraint_diag_K);
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
    cudaFree(m_data.dev_sample_to_grid_id);

    cudaFree(m_data.dev_energy);
    cudaFree(m_data.dev_residual);

    cudaFree(m_data.dev_gravity);

    cudaFree(m_dev_data);
}

template <typename Real>
void ImplicitCPICSolver<Real>::Step()
{
    ImplicitCPICSolverKernel::calc_sample_to_leftbottom_grid_id<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaMemset(m_data.dev_grid_tri_info, 0xFF, sizeof(uint64_t) * m_data.m_num_grid);
    ImplicitCPICSolverKernel::sample_to_grid<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);

    // MPM
    cudaMemset(m_data.dev_grid_momentum, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_mass, 0, sizeof(Real) * m_data.m_num_grid);
    cudaMemset(m_data.dev_grid_diag_Kv, 0, sizeof(Real) * m_data.m_num_grid * 3);
    ImplicitCPICSolverKernel::update_F<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    ImplicitCPICSolverKernel::particles_gravity<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    ImplicitCPICSolverKernel::calc_particle_to_leftbottom_grid_id<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    ImplicitCPICSolverKernel::P2G_momentum_and_mass_and_K<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    ImplicitCPICSolverKernel::calc_grids_velocity<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaMemcpy(m_data.dev_grid_velocity_hat, m_data.dev_grid_velocity, sizeof(Real) * m_data.m_num_grid * 3, cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_data.dev_grid_velocity_prev, m_data.dev_grid_velocity, sizeof(Real) * m_data.m_num_grid * 3, cudaMemcpyDeviceToDevice);

    // PD
    cudaMemcpy(m_data.dev_position_backup, m_data.dev_position, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);
    ImplicitCPICSolverKernel::initial_guess<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaMemcpy(m_data.dev_velocity_hat, m_data.dev_velocity, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_data.dev_velocity_prev, m_data.dev_velocity, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);
    
    Real omega = 1;
    for (unsigned int iter = 0; iter < MAX_ITERATIONS; ++iter)
    {
        // MPM fluid local solve
        cudaMemset(m_data.dev_grid_force, 0, sizeof(Real) * m_data.m_num_grid * 3);
        ImplicitCPICSolverKernel::G2P_calc_particle_temp_F<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
        ImplicitCPICSolverKernel::P2G_calc_grid_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
        ImplicitCPICSolverKernel::grid_op<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);

        // PD solid local solve
        cudaMemset(m_data.dev_vert_force, 0, sizeof(Real) * m_data.m_num_vert * 3);
        cudaMemset(m_data.dev_constraint_diag_K, 0, sizeof(Real) * m_data.m_num_vert * 3);
        cudaPhysics::array_axpby(m_data.dev_position, (Real)1, m_data.dev_position_backup, m_data.m_time_step, m_data.dev_velocity, 3 * m_data.m_num_vert);
        ImplicitCPICSolverKernel::calc_tetrahedron_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
        ImplicitCPICSolverKernel::accumulate_vert_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
        ImplicitCPICSolverKernel::calc_box_collision_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);

        // couple local solve
        ImplicitCPICSolverKernel::inside_grids_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);

        // global solve
        cudaMemset(m_data.dev_residual, 0, sizeof(Real) * m_data.m_num_grid * 3);
        ImplicitCPICSolverKernel::mpm_jacobi_iteration<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
        ImplicitCPICSolverKernel::pd_jacobi_iteration<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
        // if (m_verbose)
        // {
        //     Real residual = 0;
        //     cudaMemcpy(&residual, m_data.dev_residual, sizeof(Real), cudaMemcpyDeviceToHost);
        //     printf("iter %u residual = %e\n", iter, residual);
        // }

        Real alpha = 1.0f;
        if (iter % LINE_SEARCH_ITER == 0)
        {
            cudaMemcpy(m_data.dev_grid_velocity_next, m_data.dev_grid_velocity, sizeof(Real) * m_data.m_num_grid * 3, cudaMemcpyDeviceToDevice);
            cudaMemcpy(m_data.dev_velocity_next, m_data.dev_velocity, sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);
            cudaMemset(m_data.dev_energy, 0, sizeof(Real));
            Real initial_energy = 0;
            ImplicitCPICSolverKernel::calc_particle_internal_energy<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
            ImplicitCPICSolverKernel::calc_grid_energy<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
            ImplicitCPICSolverKernel::calc_tet_energy<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
            ImplicitCPICSolverKernel::calc_vert_energy<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
            ImplicitCPICSolverKernel::calc_couple_energy<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
            cudaMemcpy(&initial_energy, m_data.dev_energy, sizeof(Real), cudaMemcpyDeviceToHost);
            if (m_verbose)
                printf("  iter %u, initial energy = %e\n", iter, initial_energy);
            while (alpha > MIN_ALPHA)
            {
                cudaMemset(m_data.dev_energy, 0, sizeof(Real));
                cudaPhysics::array_axpby(m_data.dev_grid_velocity_next, (Real)1, m_data.dev_grid_velocity, alpha, m_data.dev_grid_velocity_delta, 3 * m_data.m_num_grid);
                cudaPhysics::array_axpby(m_data.dev_velocity_next, (Real)1, m_data.dev_velocity, alpha, m_data.dev_velocity_delta, 3 * m_data.m_num_vert);
                cudaPhysics::array_axpby(m_data.dev_position, (Real)1, m_data.dev_position_backup, m_data.m_time_step, m_data.dev_velocity_next, 3 * m_data.m_num_vert);
                ImplicitCPICSolverKernel::G2P_calc_particle_temp_F<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data, alpha);
                ImplicitCPICSolverKernel::calc_particle_internal_energy<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
                ImplicitCPICSolverKernel::calc_grid_energy<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
                ImplicitCPICSolverKernel::calc_tet_energy<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
                ImplicitCPICSolverKernel::calc_vert_energy<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data);
                ImplicitCPICSolverKernel::calc_couple_energy<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
                Real current_energy = 0;
                cudaMemcpy(&current_energy, m_data.dev_energy, sizeof(Real), cudaMemcpyDeviceToHost);
                if (m_verbose)
                    printf("    alpha %e, current energy = %e\n", alpha, current_energy);
                if (current_energy < initial_energy)
                    break;
                alpha *= 0.5f;
            }
        }
        cudaPhysics::array_axpby(m_data.dev_grid_velocity_next, (Real)1, m_data.dev_grid_velocity, alpha, m_data.dev_grid_velocity_delta, 3 * m_data.m_num_grid);
        cudaPhysics::array_axpby(m_data.dev_velocity_next, (Real)1, m_data.dev_velocity, alpha, m_data.dev_velocity_delta, 3 * m_data.m_num_vert);
        UpdateChebyshevOmega(omega, iter);
        ImplicitCPICSolverKernel::MPM_Chebyshev<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data, omega);
        ImplicitCPICSolverKernel::PD_Chebyshev<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data, omega);
        SwapAnswerBuffers();
    }

    ImplicitCPICSolverKernel::grids_boundary_conditions<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaMemset(m_data.dev_particle_velocity, 0, sizeof(Real) * m_data.m_num_particle * 3);
    cudaMemset(m_data.dev_particle_C, 0, sizeof(Real) * m_data.m_num_particle * 9);
    ImplicitCPICSolverKernel::G2P_velocity_and_C<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    ImplicitCPICSolverKernel::update_particle_positions<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);

    cudaPhysics::array_axpby(m_data.dev_position, (Real)1, m_data.dev_position_backup, m_data.m_time_step, m_data.dev_velocity, 3 * m_data.m_num_vert);
    ImplicitCPICSolverKernel::calc_sample_position<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_dev_data);
}

template <typename Real>
Real *ImplicitCPICSolver<Real>::GetDeviceNodePositions()
{
    return m_data.dev_position;
}

template <typename Real>
Real *ImplicitCPICSolver<Real>::GetDeviceSamplePositions()
{
    return m_data.dev_sample_position;
}

template <typename Real>
Real *ImplicitCPICSolver<Real>::GetDeviceParticlePositions()
{
    return m_data.dev_particle_position;
}

template <typename Real>
void ImplicitCPICSolver<Real>::UpdateChebyshevOmega(Real &omega, unsigned int iter)
{
    if (iter < CHEBYSHEV_DELAY_ITER)
        omega = 1;
    else if (iter == CHEBYSHEV_DELAY_ITER)
        omega = 2 / (2 - CHEBYSHEV_RHO * CHEBYSHEV_RHO);
    else
        omega = 4 / (4 - CHEBYSHEV_RHO * CHEBYSHEV_RHO * omega);
}

namespace ImplicitCPICSolverKernel
{
    template <typename Real>
    __global__ void swap_answer(ImplicitCPICSolverData<Real> *data)
    {
        Real *temp;
        temp = data->dev_velocity;
        data->dev_velocity = data->dev_velocity_prev;
        data->dev_velocity_prev = temp;
        temp = data->dev_velocity;
        data->dev_velocity = data->dev_velocity_next;
        data->dev_velocity_next = temp;

        temp = data->dev_grid_velocity;
        data->dev_grid_velocity = data->dev_grid_velocity_prev;
        data->dev_grid_velocity_prev = temp;
        temp = data->dev_grid_velocity;
        data->dev_grid_velocity = data->dev_grid_velocity_next;
        data->dev_grid_velocity_next = temp;
    }
}

template <typename Real>
void ImplicitCPICSolver<Real>::SwapAnswerBuffers()
{
    std::swap(m_data.dev_velocity, m_data.dev_velocity_prev);
    std::swap(m_data.dev_velocity, m_data.dev_velocity_next);
    std::swap(m_data.dev_grid_velocity, m_data.dev_grid_velocity_prev);
    std::swap(m_data.dev_grid_velocity, m_data.dev_grid_velocity_next);
    ImplicitCPICSolverKernel::swap_answer<Real><<<1, 1>>>(m_dev_data);
}

template class ImplicitCPICSolver<float>;
template class ImplicitCPICSolver<double>;
template struct ImplicitCPICSolverData<float>;
template struct ImplicitCPICSolverData<double>;

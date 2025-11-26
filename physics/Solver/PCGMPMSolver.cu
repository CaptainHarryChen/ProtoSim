#include "PCGMPMSolver.cuh"
#include <cuda_utils/error.cuh>
#include <cuda_utils/array.cuh>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <Math/algebra.cuh>
#include <Math/ConstitutiveModel/Neohookean.cuh>

namespace PCGMPMSolverKernel
{
    template <typename Real>
    __global__ void update_F(const PCGMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        assert(data->dev_particle_type[i] == MPM_FLUID);
        {
            // Only use the first element of F to store J
            Real J = data->dev_particle_F[i * 9] * (1 + data->m_time_step * (data->dev_particle_C[i * 9] + data->dev_particle_C[i * 9 + 4] + data->dev_particle_C[i * 9 + 8]));
            data->dev_particle_F[i * 9 + 0] = J;
        }
    }

    template <typename Real>
    __global__ void calc_particle_to_leftbottom_grid_id(const PCGMPMSolverData<Real> *data)
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
            x = max(0, min(x, (int)data->dev_grid_size[0] - 3));
            y = max(0, min(y, (int)data->dev_grid_size[1] - 3));
            z = max(0, min(z, (int)data->dev_grid_size[2] - 3));
            printf("Warning: particle %d is out of grid, set to %d %d %d [Particle position] %.10f %.10f %.10f\n", i, x, y, z, data->dev_particle_position[i * 3 + 0], data->dev_particle_position[i * 3 + 1], data->dev_particle_position[i * 3 + 2]);
        }

        data->dev_particle_to_grid_id[i] = x * data->dev_grid_size[1] * data->dev_grid_size[2] + y * data->dev_grid_size[2] + z;
    }

    template <typename Real>
    __device__ void get_grid_position(Real *grid_position, unsigned int id, const PCGMPMSolverData<Real> *data)
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
    __global__ void P2G_momentum_and_mass_and_B_const(const PCGMPMSolverData<Real> *data)
    {
        unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= data->m_num_particle)
            return;
        Real coef = - data->m_fluid_lambda 
                    * data->m_time_step
                    * 16 / (data->m_grid_spacing * data->m_grid_spacing * data->m_grid_spacing * data->m_grid_spacing)
                    * data->dev_particle_volume[p]
                    * (data->dev_particle_F[p * 9] * data->dev_particle_F[p * 9]);
        // viscosity
        coef += - data->m_fluid_viscosity
                 * 64 / (3 * data->m_grid_spacing * data->m_grid_spacing * data->m_grid_spacing * data->m_grid_spacing)
                 * data->dev_particle_volume[p] * data->dev_particle_F[p * 9];

        const Real *particle_position = &data->dev_particle_position[p * 3];
        const Real *particle_velocity = &data->dev_particle_velocity[p * 3];
        unsigned int leftbottom_grid_id = data->dev_particle_to_grid_id[p];
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
                    Real weight = grid_particle_quadratic_weight(grid_position, particle_position, data->m_grid_spacing);
                    Real momentum[3];
                    Real delta_position[3];
                    cudaPhysics::vecSubs3(delta_position, grid_position, particle_position);
                    cudaPhysics::matVec3(momentum, &data->dev_particle_C[p * 9], delta_position);
                    cudaPhysics::vecMul3(momentum, data->dev_particle_mass[p], momentum);

                    Real temp_momentum[3];
                    cudaPhysics::vecMul3(temp_momentum, data->dev_particle_mass[p], particle_velocity);
                    cudaPhysics::vecAdd3(momentum, temp_momentum, momentum);
                    cudaPhysics::vecMul3(momentum, weight, momentum);
                    atomicAdd(&data->dev_grid_momentum[grid_id * 3 + 0], momentum[0]);
                    atomicAdd(&data->dev_grid_momentum[grid_id * 3 + 1], momentum[1]);
                    atomicAdd(&data->dev_grid_momentum[grid_id * 3 + 2], momentum[2]);

                    atomicAdd(&data->dev_grid_mass[grid_id], weight * data->dev_particle_mass[p]);

                    Real diag_K[3];
                    cudaPhysics::vecMul3(diag_K, delta_position, delta_position);
                    cudaPhysics::vecMul3(diag_K, coef * weight * weight);
                    atomicAdd(&data->dev_grid_diag_B_const[grid_id * 3 + 0], diag_K[0]); // the B_const is not B for now, it stores K here
                    atomicAdd(&data->dev_grid_diag_B_const[grid_id * 3 + 1], diag_K[1]);
                    atomicAdd(&data->dev_grid_diag_B_const[grid_id * 3 + 2], diag_K[2]);
                }
    }

    template <typename Real>
    __global__ void grid_preprocess(const PCGMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        if (data->dev_grid_mass[i] > 0)
        {
            // calculate grid velocity
            cudaPhysics::vecMul3(&data->dev_grid_velocity[i * 3], (Real)(1. / data->dev_grid_mass[i]), &data->dev_grid_momentum[i * 3]);
            // calculate the const term (identity and elastic) of preconditioner diag_B
            for (unsigned int j = 0; j < 3; ++j)
                data->dev_grid_diag_B_const[i * 3 + j] = 1 - data->m_time_step / data->dev_grid_mass[i] * data->dev_grid_diag_B_const[i * 3 + j];
        }
        else
        {
            data->dev_grid_velocity[i * 3 + 0] = 0;
            data->dev_grid_velocity[i * 3 + 1] = 0;
            data->dev_grid_velocity[i * 3 + 2] = 0;
        }
    }

    template <typename Real>
    __global__ void grids_gravity(const PCGMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        Real delta_velocity[3];
        cudaPhysics::vecMul3(delta_velocity, data->m_time_step, data->dev_gravity);
        cudaPhysics::vecAdd3(&data->dev_grid_velocity[i * 3], &data->dev_grid_velocity[i * 3], delta_velocity);
    }

    template <typename Real>
    __global__ void particles_gravity(const PCGMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        Real delta_velocity[3];
        cudaPhysics::vecMul3(delta_velocity, data->m_time_step, data->dev_gravity);
        cudaPhysics::vecAdd3(&data->dev_particle_velocity[i * 3], &data->dev_particle_velocity[i * 3], delta_velocity);
    }

    template <typename Real>
    __global__ void G2P_calc_particle_temp_C(const PCGMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        Real temp_sum[9] = {0};
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
                    get_grid_position(grid_position, grid_id, data);
                    Real weight = grid_particle_quadratic_weight(grid_position, particle_position, data->m_grid_spacing);

                    Real delta_position[3], temp_vecvec[9];
                    cudaPhysics::vecSubs3(delta_position, grid_position, particle_position);
                    cudaPhysics::vecvecT(temp_vecvec, &data->dev_grid_velocity[grid_id * 3], delta_position, 3, 3);
                    cudaPhysics::axpby(temp_sum, (Real)1, temp_sum, weight, temp_vecvec, 9);
                }
        cudaPhysics::vecMul(&data->dev_particle_temp_C[i * 9], 4 / (data->m_grid_spacing * data->m_grid_spacing), temp_sum, 9);
    }

    template <typename Real>
    __global__ void P2G_calc_grid_force(const PCGMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        const Real *particle_position = &data->dev_particle_position[i * 3];
        const Real *particle_velocity = &data->dev_particle_velocity[i * 3];
        Real tr_C = data->dev_particle_temp_C[i * 9] + data->dev_particle_temp_C[i * 9 + 4] + data->dev_particle_temp_C[i * 9 + 8];
        Real temp_J = data->dev_particle_F[i * 9] * (1 + data->m_time_step * tr_C);
        Real elastic_force_coef = - data->dev_particle_volume[i] 
                                  * data->m_fluid_lambda * (temp_J - 1)
                                  * data->dev_particle_F[i * 9]
                                  * 4 / (data->m_grid_spacing * data->m_grid_spacing);
        Real viscosity_force_coef[9] = {0};
        viscosity_force_coef[0] = viscosity_force_coef[4] = viscosity_force_coef[8] = - 0.66666667f * tr_C;
        cudaPhysics::axpby(viscosity_force_coef, (Real)1, viscosity_force_coef, (Real)2, &data->dev_particle_temp_C[i * 9], 9);
        cudaPhysics::vecMul(viscosity_force_coef, 
                            - data->m_fluid_viscosity * data->dev_particle_volume[i] * data->dev_particle_F[i * 9] * 4 / (data->m_grid_spacing * data->m_grid_spacing), 
                            viscosity_force_coef, 
                            9);
        
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
                    get_grid_position(grid_position, grid_id, data);
                    Real weight = grid_particle_quadratic_weight(grid_position, particle_position, data->m_grid_spacing);
                    Real delta_position[3];
                    cudaPhysics::vecSubs3(delta_position, grid_position, particle_position);
                    Real elastic_force[3], viscosity_force[3];
                    cudaPhysics::vecMul3(elastic_force, elastic_force_coef * weight, delta_position);
                    cudaPhysics::matVec3(viscosity_force, viscosity_force_coef, delta_position);
                    cudaPhysics::vecMul3(viscosity_force, weight, viscosity_force);
                    atomicAdd(&data->dev_grid_force[grid_id * 3 + 0], elastic_force[0] + viscosity_force[0]);
                    atomicAdd(&data->dev_grid_force[grid_id * 3 + 1], elastic_force[1] + viscosity_force[1]);
                    atomicAdd(&data->dev_grid_force[grid_id * 3 + 2], elastic_force[2] + viscosity_force[2]);
                }
    }

    template <typename Real>
    __global__ void grid_boundary_force(const PCGMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        // apply ground boundary condition
        unsigned int x = i / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (i / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = i % data->dev_grid_size[2];
        Real grid_pos[3];
        get_grid_position(grid_pos, i, data);
        cudaPhysics::axpby(grid_pos, (Real)1, grid_pos, data->m_time_step, &data->dev_grid_velocity[i * 3], 3);
        Real *diag_B = &data->dev_grid_diag_B_mutable[i * 3];
        Real *grid_force = &data->dev_grid_force[i * 3];
        for (unsigned int j = 0; j < 3; j++)
        {
            if (grid_pos[j] <= data->dev_inner_bbox[j])
            {
                diag_B[j] += data->m_ground_stiffness * data->m_time_step * data->m_time_step;
                grid_force[j] += data->dev_grid_mass[i] * data->m_ground_stiffness * (data->dev_inner_bbox[j] - grid_pos[j]);
            }
            if (grid_pos[j] >= data->dev_inner_bbox[j + 3])
            {
                diag_B[j] += data->m_ground_stiffness * data->m_time_step * data->m_time_step;
                grid_force[j] += data->dev_grid_mass[i] * data->m_ground_stiffness * (data->dev_inner_bbox[j + 3] - grid_pos[j]);
            }
        }
    }

    template <typename Real>
    __global__ void grid_r_dot_r(const PCGMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            Real r = data->dev_grid_velocity[i * 3 + j] - data->dev_grid_velocity_hat[i * 3 + j] - data->m_time_step / data->dev_grid_mass[i] * data->dev_grid_force[i * 3 + j];
            data->dev_grid_temp[i * 3 + j] = r * r;
        }
        if (data->dev_grid_mass[i] <= 0)
        {
            data->dev_grid_temp[i * 3 + 0] = 0;
            data->dev_grid_temp[i * 3 + 1] = 0;
            data->dev_grid_temp[i * 3 + 2] = 0;
        }
    }

    template <typename Real>
    __global__ void grid_z_dot_r(const PCGMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            Real r = data->dev_grid_velocity[i * 3 + j] - data->dev_grid_velocity_hat[i * 3 + j] - data->m_time_step / data->dev_grid_mass[i] * data->dev_grid_force[i * 3 + j];
            data->dev_grid_temp[i * 3 + j] = r * r / (data->dev_grid_diag_B_const[i * 3 + j] + data->dev_grid_diag_B_mutable[i * 3 + j]);
        }
        if (data->dev_grid_mass[i] <= 0)
        {
            data->dev_grid_temp[i * 3 + 0] = 0;
            data->dev_grid_temp[i * 3 + 1] = 0;
            data->dev_grid_temp[i * 3 + 2] = 0;
        }
    }

    template <typename Real>
    __global__ void grid_search_direction(const PCGMPMSolverData<Real> *data, Real beta)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            Real r = data->dev_grid_velocity[i * 3 + j] - data->dev_grid_velocity_hat[i * 3 + j] - data->m_time_step / data->dev_grid_mass[i] * data->dev_grid_force[i * 3 + j];
            data->dev_grid_p[i * 3 + j] = - r / (data->dev_grid_diag_B_const[i * 3 + j] + data->dev_grid_diag_B_mutable[i * 3 + j])
                                          + beta * data->dev_grid_p[i * 3 + j];
        }
        if (data->dev_grid_mass[i] <= 0)
        {
            data->dev_grid_p[i * 3 + 0] = 0;
            data->dev_grid_p[i * 3 + 1] = 0;
            data->dev_grid_p[i * 3 + 2] = 0;
        }
    }

    template <typename Real>
    __global__ void G2P_calc_Ap_step1(const PCGMPMSolverData<Real> *data)
    {
        unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= data->m_num_particle)
            return;
        Real &res = data->dev_particle_temp[p];
        res = 0;
        const Real *particle_position = &data->dev_particle_position[p * 3];
        unsigned int leftbottom_grid_id = data->dev_particle_to_grid_id[p];
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
                    Real weight = grid_particle_quadratic_weight(grid_position, particle_position, data->m_grid_spacing);

                    Real delta_position[3];
                    cudaPhysics::vecSubs3(delta_position, grid_position, particle_position);
                    res += weight * cudaPhysics::dot3(&data->dev_grid_p[grid_id * 3], delta_position);
                }
    }

    template <typename Real>
    __global__ void P2G_calc_Ap_step2(const PCGMPMSolverData<Real> *data)
    {
        unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= data->m_num_particle)
            return;
        // elastic
        Real coef = - data->m_fluid_lambda 
                    * data->m_time_step
                    * 16 / (data->m_grid_spacing * data->m_grid_spacing * data->m_grid_spacing * data->m_grid_spacing)
                    * data->dev_particle_volume[p]
                    * (data->dev_particle_F[p * 9] * data->dev_particle_F[p * 9])
                    * data->dev_particle_temp[p];
        // viscosity
        coef += - data->m_fluid_viscosity
                 * 64 / (3 * data->m_grid_spacing * data->m_grid_spacing * data->m_grid_spacing * data->m_grid_spacing)
                 * data->dev_particle_volume[p] * data->dev_particle_F[p * 9]
                 * data->dev_particle_temp[p];
        
        const Real *particle_position = &data->dev_particle_position[p * 3];
        unsigned int leftbottom_grid_id = data->dev_particle_to_grid_id[p];
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
                    Real weight = grid_particle_quadratic_weight(grid_position, particle_position, data->m_grid_spacing);

                    Real delta_position[3];
                    cudaPhysics::vecSubs3(delta_position, grid_position, particle_position);
                    atomicAdd(&data->dev_grid_temp[grid_id * 3 + 0], coef * weight * delta_position[0]);
                    atomicAdd(&data->dev_grid_temp[grid_id * 3 + 1], coef * weight * delta_position[1]);
                    atomicAdd(&data->dev_grid_temp[grid_id * 3 + 2], coef * weight * delta_position[2]);
                }
    }

    template <typename Real>
    __global__ void grid_calc_pAp_step3(const PCGMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        Real coef = -data->m_time_step / data->dev_grid_mass[i];
        Real Ap_non_diag[3], Ap_diag[3];
        // elastic force part
        cudaPhysics::vecMul3(Ap_non_diag, coef, &data->dev_grid_temp[i * 3]);
        // boundary force part
        cudaPhysics::vecMul3(Ap_diag, &data->dev_grid_diag_B_mutable[i * 3], &data->dev_grid_p[i * 3]);
        // total Ap
        cudaPhysics::axpbypcz(&data->dev_grid_temp[i * 3], (Real)1, &data->dev_grid_p[i * 3], (Real)1, Ap_non_diag, (Real)1, Ap_diag, 3);
        // pAp
        cudaPhysics::vecMul3(&data->dev_grid_temp[i * 3], &data->dev_grid_p[i * 3], &data->dev_grid_temp[i * 3]);
        if (data->dev_grid_mass[i] <= 0)
        {
            data->dev_grid_temp[i * 3 + 0] = 0;
            data->dev_grid_temp[i * 3 + 1] = 0;
            data->dev_grid_temp[i * 3 + 2] = 0;
        }
    }

    template <typename Real>
    __global__ void grid_p_dot_r(const PCGMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            Real r = data->dev_grid_velocity[i * 3 + j] - data->dev_grid_velocity_hat[i * 3 + j] - data->m_time_step / data->dev_grid_mass[i] * data->dev_grid_force[i * 3 + j];
            data->dev_grid_temp[i * 3 + j] = data->dev_grid_p[i * 3 + j] * r;
        }
        if (data->dev_grid_mass[i] <= 0)
        {
            data->dev_grid_temp[i * 3 + 0] = 0;
            data->dev_grid_temp[i * 3 + 1] = 0;
            data->dev_grid_temp[i * 3 + 2] = 0;
        }
    }

    template <typename Real>
    __global__ void grids_boundary_conditions(PCGMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        unsigned int x = i / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (i / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = i % data->dev_grid_size[2];
        Real grid_pos[3];
        unsigned int grid_id = x * data->dev_grid_size[1] * data->dev_grid_size[2] + y * data->dev_grid_size[2] + z;
        get_grid_position(grid_pos, grid_id, data);
        for (unsigned int j = 0; j < 3; j++)
        {
            if (grid_pos[j] + data->dev_grid_velocity[i * 3 + j] * data->m_time_step < data->dev_inner_bbox[j])
            {
                data->dev_grid_velocity[i * 3 + j] = (data->dev_inner_bbox[j] - grid_pos[j]) / data->m_time_step;
            }
            if (grid_pos[j] + data->dev_grid_velocity[i * 3 + j] * data->m_time_step > data->dev_inner_bbox[j + 3])
            {
                data->dev_grid_velocity[i * 3 + j] = (data->dev_inner_bbox[j + 3] - grid_pos[j]) / data->m_time_step;
            }
        }
    }

    template <typename Real>
    __global__ void G2P_velocity_and_C(const PCGMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
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
                    get_grid_position(grid_position, grid_id, data);
                    Real weight = grid_particle_quadratic_weight(grid_position, particle_position, data->m_grid_spacing);

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
    __global__ void update_particle_positions(const PCGMPMSolverData<Real> *data)
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
    __global__ void particles_boundary_conditions(const PCGMPMSolverData<Real> *data)
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
}

template <typename Real>
PCGMPMSolver<Real>::PCGMPMSolver(
    const std::vector<Real> &particle_position, const std::vector<unsigned int> &particle_type, const std::vector<Real> &particle_mass, const std::vector<Real> &particle_volume,
    std::vector<Real> bbox, Real grid_spacing, unsigned int boundary_thickness)
{
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
    cudaMalloc(&m_data.dev_particle_temp_C, sizeof(Real) * m_data.m_num_particle * 9);
    cudaMalloc(&m_data.dev_particle_F, sizeof(Real) * m_data.m_num_particle * 9);
    cudaPhysics::fill_identity_matrix(m_data.dev_particle_F, m_data.m_num_particle, 3);
    cudaMalloc(&m_data.dev_particle_to_grid_id, sizeof(unsigned int) * m_data.m_num_particle);
    cudaMalloc(&m_data.dev_particle_temp, sizeof(Real) * m_data.m_num_particle);

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
    cudaMemset(m_data.dev_grid_force, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_velocity, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_velocity, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_velocity_hat, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_velocity_hat, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_diag_B_const, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_diag_B_const, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_diag_B_mutable, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_diag_B_mutable, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_p, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_p, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_temp, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_temp, 0, sizeof(Real) * m_data.m_num_grid * 3);

    m_data.m_time_step = TIME_STEP;
    m_data.m_ground_stiffness = GROUND_COLLISION_STIFFNESS;
    m_data.m_fluid_lambda = FLUID_LAMBDA;
    m_data.m_fluid_viscosity = FLUID_VISCOSITY;
    cudaMalloc(&m_data.dev_gravity, sizeof(Real) * 3);
    std::vector<Real> gravity = {0., -GRAVITY, 0.};
    cudaMemcpy(m_data.dev_gravity, gravity.data(), sizeof(Real) * 3, cudaMemcpyHostToDevice);

    cudaMalloc(&m_dev_data, sizeof(PCGMPMSolverData<Real>));
    cudaMemcpy(m_dev_data, &m_data, sizeof(PCGMPMSolverData<Real>), cudaMemcpyHostToDevice);
}

template <typename Real>
PCGMPMSolver<Real>::~PCGMPMSolver()
{
    cudaFree(m_data.dev_particle_position);
    cudaFree(m_data.dev_particle_velocity);
    cudaFree(m_data.dev_particle_mass);
    cudaFree(m_data.dev_particle_volume);
    cudaFree(m_data.dev_particle_type);
    cudaFree(m_data.dev_particle_C);
    cudaFree(m_data.dev_particle_temp_C);
    cudaFree(m_data.dev_particle_F);
    cudaFree(m_data.dev_particle_to_grid_id);
    cudaFree(m_data.dev_particle_temp);
    cudaFree(m_data.dev_inner_bbox);
    cudaFree(m_data.dev_outer_bbox);
    cudaFree(m_data.dev_grid_size);
    cudaFree(m_data.dev_grid_momentum);
    cudaFree(m_data.dev_grid_mass);
    cudaFree(m_data.dev_grid_force);
    cudaFree(m_data.dev_grid_velocity);
    cudaFree(m_data.dev_grid_velocity_hat);
    cudaFree(m_data.dev_grid_diag_B_const);
    cudaFree(m_data.dev_grid_diag_B_mutable);
    cudaFree(m_data.dev_grid_p);
    cudaFree(m_data.dev_grid_temp);

    cudaFree(m_data.dev_gravity);

    cudaFree(m_dev_data);
}

template <typename Real>
void PCGMPMSolver<Real>::Step()
{
    // P2G
    cudaMemset(m_data.dev_grid_momentum, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_mass, 0, sizeof(Real) * m_data.m_num_grid);
    cudaMemset(m_data.dev_grid_diag_B_const, 0, sizeof(Real) * m_data.m_num_grid * 3);
    PCGMPMSolverKernel::particles_gravity<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    PCGMPMSolverKernel::calc_particle_to_leftbottom_grid_id<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    PCGMPMSolverKernel::P2G_momentum_and_mass_and_B_const<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    PCGMPMSolverKernel::grid_preprocess<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaMemcpy(m_data.dev_grid_velocity_hat, m_data.dev_grid_velocity, sizeof(Real) * m_data.m_num_grid * 3, cudaMemcpyDeviceToDevice);

    PCGSolver();

    cudaMemset(m_data.dev_particle_velocity, 0, sizeof(Real) * m_data.m_num_particle * 3);
    cudaMemset(m_data.dev_particle_C, 0, sizeof(Real) * m_data.m_num_particle * 9);
    PCGMPMSolverKernel::G2P_velocity_and_C<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    PCGMPMSolverKernel::update_F<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    PCGMPMSolverKernel::update_particle_positions<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    // PCGMPMSolverKernel::particles_boundary_conditions<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
}

template <typename Real>
Real *PCGMPMSolver<Real>::GetDevicePositions()
{
    return m_data.dev_particle_position;
}

template <typename Real>
void PCGMPMSolver<Real>::PCGSolver()
{
    Real prev_z_dot_r = 0; // for calculate beta in PCG
    for (unsigned int iter = 0; iter < MAX_ITERATIONS; ++iter)
    {
        if (m_verbose)
            printf("PCG iteration %u\n", iter);

        cudaMemset(m_data.dev_grid_force, 0, sizeof(Real) * m_data.m_num_grid * 3);
        cudaMemset(m_data.dev_grid_diag_B_mutable, 0, sizeof(Real) * m_data.m_num_grid * 3);
        PCGMPMSolverKernel::G2P_calc_particle_temp_C<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
        PCGMPMSolverKernel::P2G_calc_grid_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
        PCGMPMSolverKernel::grid_boundary_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);

        PCGMPMSolverKernel::grid_r_dot_r<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
        Real residual = thrust::reduce(thrust::device_pointer_cast(m_data.dev_grid_temp),
                                       thrust::device_pointer_cast(m_data.dev_grid_temp + m_data.m_num_grid * 3)) / m_data.m_num_grid;
        if (m_verbose)
            printf("  residual = %e\n", residual);
        if (residual < RESIDUAL_TOLERANCE)
            break;
        
        Real beta = 0;
        PCGMPMSolverKernel::grid_z_dot_r<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
        Real z_dot_r = thrust::reduce(thrust::device_pointer_cast(m_data.dev_grid_temp),
                                      thrust::device_pointer_cast(m_data.dev_grid_temp + m_data.m_num_grid * 3));
        if (iter > 0)
            beta = z_dot_r / prev_z_dot_r;
        prev_z_dot_r = z_dot_r;
        PCGMPMSolverKernel::grid_search_direction<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data, beta);

        thrust::transform(thrust::device_pointer_cast(m_data.dev_grid_p),
                          thrust::device_pointer_cast(m_data.dev_grid_p + m_data.m_num_grid * 3),
                          thrust::device_pointer_cast(m_data.dev_grid_temp),
                          thrust::placeholders::_1 * thrust::placeholders::_1);
        Real p_dot_p = thrust::reduce(thrust::device_pointer_cast(m_data.dev_grid_temp),
                                      thrust::device_pointer_cast(m_data.dev_grid_temp + m_data.m_num_grid * 3));
        thrust::transform(thrust::device_pointer_cast(m_data.dev_grid_p),
                          thrust::device_pointer_cast(m_data.dev_grid_p + m_data.m_num_grid * 3),
                          thrust::device_pointer_cast(m_data.dev_grid_temp),
                          thrust::placeholders::_1 / sqrt(p_dot_p));
        cudaMemcpy(m_data.dev_grid_p, m_data.dev_grid_temp, sizeof(Real) * m_data.m_num_grid * 3, cudaMemcpyDeviceToDevice);
    
        PCGMPMSolverKernel::G2P_calc_Ap_step1<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
        cudaMemset(m_data.dev_grid_temp, 0, sizeof(Real) * m_data.m_num_grid * 3);
        PCGMPMSolverKernel::P2G_calc_Ap_step2<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
        PCGMPMSolverKernel::grid_calc_pAp_step3<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
        Real pAp = thrust::reduce(thrust::device_pointer_cast(m_data.dev_grid_temp),
                                  thrust::device_pointer_cast(m_data.dev_grid_temp + m_data.m_num_grid * 3));
        PCGMPMSolverKernel::grid_p_dot_r<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
        Real alpha = - thrust::reduce(thrust::device_pointer_cast(m_data.dev_grid_temp),
                                      thrust::device_pointer_cast(m_data.dev_grid_temp + m_data.m_num_grid * 3)) / pAp;

        thrust::transform(thrust::device_pointer_cast(m_data.dev_grid_velocity),
                          thrust::device_pointer_cast(m_data.dev_grid_velocity + m_data.m_num_grid * 3),
                          thrust::device_pointer_cast(m_data.dev_grid_p),
                          thrust::device_pointer_cast(m_data.dev_grid_velocity),
                          thrust::placeholders::_1 + alpha * thrust::placeholders::_2);
    }
    PCGMPMSolverKernel::grids_boundary_conditions<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
}

template class PCGMPMSolver<float>;
template class PCGMPMSolver<double>;
template struct PCGMPMSolverData<float>;
template struct PCGMPMSolverData<double>;

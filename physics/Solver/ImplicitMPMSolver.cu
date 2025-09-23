#include "ImplicitMPMSolver.cuh"
#include <cuda_utils/error.cuh>
#include <cuda_utils/array.cuh>
#include <Math/algebra.cuh>
#include <Math/elastic_model.cuh>

namespace ImplicitMPMSolverKernel
{
    template <typename Real>
    __global__ void update_F(ImplicitMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        if (data->dev_particle_type[i] == MPM_ELASTIC)
        {
            // Real F[9], temp[9];
            // cudaPhysics::matMul3(temp, data->m_time_step, &data->dev_particle_C[i * 9]);
            // temp[0] += 1.;
            // temp[4] += 1.;
            // temp[8] += 1.;
            // cudaPhysics::matMul3(F, temp, &data->dev_particle_F[i * 9]);
            // cudaPhysics::vecCopy(&data->dev_particle_F[i * 9], F, 9);
        }
        else if (data->dev_particle_type[i] == MPM_FLUID)
        {
            // Only use the first element of F to store J
            Real J = data->dev_particle_F[i * 9] * (1 + data->m_time_step * (data->dev_particle_C[i * 9] + data->dev_particle_C[i * 9 + 4] + data->dev_particle_C[i * 9 + 8]));
            data->dev_particle_F[i * 9 + 0] = J;
        }
    }

    template <typename Real>
    __global__ void calc_particle_to_leftbottom_grid_id(ImplicitMPMSolverData<Real> *data)
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
    __device__ void get_grid_position(Real *grid_position, unsigned int id, ImplicitMPMSolverData<Real> *data)
    {
        unsigned int z = id % data->dev_grid_size[2];
        unsigned int y = (id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int x = id / data->dev_grid_size[2] / data->dev_grid_size[1];
        grid_position[0] = data->dev_outer_bbox[0] + x * data->m_grid_spacing;
        grid_position[1] = data->dev_outer_bbox[1] + y * data->m_grid_spacing;
        grid_position[2] = data->dev_outer_bbox[2] + z * data->m_grid_spacing;
    }

    template <typename Real>
    __device__ Real grid_particle_quadratic_weight(Real *grid_position, Real *particle_position, Real grid_spacing)
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
    __global__ void P2G_momentum_and_mass_and_K(ImplicitMPMSolverData<Real> *data)
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
                    get_grid_position(grid_position, grid_id, data);
                    Real weight = grid_particle_quadratic_weight(grid_position, particle_position, data->m_grid_spacing);
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
                    atomicAdd(&data->dev_grid_diag_K[grid_id * 3 + 0], diag_K[0]);
                    atomicAdd(&data->dev_grid_diag_K[grid_id * 3 + 1], diag_K[1]);
                    atomicAdd(&data->dev_grid_diag_K[grid_id * 3 + 2], diag_K[2]);
                }
    }

    template <typename Real>
    __global__ void calc_grids_velocity(ImplicitMPMSolverData<Real> *data)
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
    __global__ void grids_gravity(ImplicitMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        Real delta_velocity[3];
        cudaPhysics::vecMul3(delta_velocity, data->m_time_step, data->dev_gravity);
        cudaPhysics::vecAdd3(&data->dev_grid_velocity[i * 3], &data->dev_grid_velocity[i * 3], delta_velocity);
    }

    template <typename Real>
    __global__ void particles_gravity(ImplicitMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        Real delta_velocity[3];
        cudaPhysics::vecMul3(delta_velocity, data->m_time_step, data->dev_gravity);
        cudaPhysics::vecAdd3(&data->dev_particle_velocity[i * 3], &data->dev_particle_velocity[i * 3], delta_velocity);
    }

    template <typename Real>
    __global__ void G2P_calc_particle_temp_F(ImplicitMPMSolverData<Real> *data, Real alpha = 0)
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
                    get_grid_position(grid_position, grid_id, data);
                    Real weight = grid_particle_quadratic_weight(grid_position, particle_position, data->m_grid_spacing);

                    Real delta_position[3], vel[3];
                    cudaPhysics::vecSubs3(delta_position, grid_position, particle_position);
                    cudaPhysics::axpby(vel, (Real)1, &data->dev_grid_velocity[grid_id * 3], alpha, &data->dev_grid_velocity_delta[grid_id * 3], 3);
                    temp_J += weight * cudaPhysics::dot3(vel, delta_position);
                }
        temp_J = (1 + data->m_time_step * 4 / (data->m_grid_spacing * data->m_grid_spacing) * temp_J) * data->dev_particle_F[i * 9];
    }

    template <typename Real>
    __global__ void P2G_calc_grid_force(ImplicitMPMSolverData<Real> *data)
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
                    get_grid_position(grid_position, grid_id, data);
                    Real weight = grid_particle_quadratic_weight(grid_position, particle_position, data->m_grid_spacing);
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
    __global__ void grid_op(ImplicitMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        Real *diag_B = &data->dev_grid_diag_B[i * 3];
        Real *grid_force = &data->dev_grid_force[i * 3];
        diag_B[0] = diag_B[1] = diag_B[2] = (Real)1;
        if (data->dev_grid_mass[i] <= 0)
            return;
        cudaPhysics::axpby(diag_B, (Real)1, diag_B, -data->m_time_step / data->dev_grid_mass[i], &data->dev_grid_diag_K[i * 3], 3);

        unsigned int x = i / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (i / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = i % data->dev_grid_size[2];
        Real *vel = &data->dev_grid_velocity[i * 3];
        if ((x <= data->m_boundary_thickness && vel[0] < 0) || (x >= data->dev_grid_size[0] - 1 - data->m_boundary_thickness && vel[0] > 0))
        {
            diag_B[0] += data->dev_grid_mass[i] * data->m_ground_stiffness * data->m_time_step;
            grid_force[0] += data->dev_grid_mass[i] * data->m_ground_stiffness * (-vel[0]) * data->m_time_step;
        }
        if ((y <= data->m_boundary_thickness && vel[1] < 0) || (y >= data->dev_grid_size[1] - 1 - data->m_boundary_thickness && vel[1] > 0))
        {
            diag_B[1] += data->dev_grid_mass[i] * data->m_ground_stiffness * data->m_time_step;
            grid_force[1] += data->dev_grid_mass[i] * data->m_ground_stiffness * (-vel[1]) * data->m_time_step;
        }
        if ((z <= data->m_boundary_thickness && vel[2] < 0) || (z >= data->dev_grid_size[2] - 1 - data->m_boundary_thickness && vel[2] > 0))
        {
            diag_B[2] += data->dev_grid_mass[i] * data->m_ground_stiffness * data->m_time_step;
            grid_force[2] += data->dev_grid_mass[i] * data->m_ground_stiffness * (-vel[2]) * data->m_time_step;
        }
    }

    template <typename Real>
    __global__ void jacobi_iteration(ImplicitMPMSolverData<Real> *data)
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
        atomicAdd(data->u_residual, grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]);
        Real inv_B[3];
        inv_B[0] = (Real)1.0 / data->dev_grid_diag_B[i * 3 + 0];
        inv_B[1] = (Real)1.0 / data->dev_grid_diag_B[i * 3 + 1];
        inv_B[2] = (Real)1.0 / data->dev_grid_diag_B[i * 3 + 2];
        cudaPhysics::vecMul3(&data->dev_grid_velocity_delta[3 * i], inv_B, grad);
    }

    template <typename Real>
    __global__ void calc_particle_internal_energy(ImplicitMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        atomicAdd(data->u_energy, 0.5 * data->m_lame_lambda * (data->dev_particle_temp_J[i] - 1) * (data->dev_particle_temp_J[i] - 1) * data->dev_particle_volume[i]);
    }

    template <typename Real>
    __global__ void calc_grid_op_energy(ImplicitMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        Real energy = 0;
        unsigned int x = i / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (i / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = i % data->dev_grid_size[2];
        Real *vel = &data->dev_grid_velocity[i * 3];
        if ((x <= data->m_boundary_thickness && vel[0] < 0) || (x >= data->dev_grid_size[0] - 1 - data->m_boundary_thickness && vel[0] > 0))
        {
            energy += 0.5 * data->m_ground_stiffness * data->dev_grid_mass[i] * vel[0] * vel[0] * data->m_time_step;
        }
        if ((y <= data->m_boundary_thickness && vel[1] < 0) || (y >= data->dev_grid_size[1] - 1 - data->m_boundary_thickness && vel[1] > 0))
        {
            energy += 0.5 * data->m_ground_stiffness * data->dev_grid_mass[i] * vel[1] * vel[1] * data->m_time_step;
        }
        if ((z <= data->m_boundary_thickness && vel[2] < 0) || (z >= data->dev_grid_size[2] - 1 - data->m_boundary_thickness && vel[2] > 0))
        {
            energy += 0.5 * data->m_ground_stiffness * data->dev_grid_mass[i] * vel[2] * vel[2] * data->m_time_step;
        }
        atomicAdd(data->u_energy, energy);
    }

    template <typename Real>
    __global__ void grids_boundary_conditions(ImplicitMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        unsigned int x = i / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (i / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = i % data->dev_grid_size[2];
        if (x <= data->m_boundary_thickness)
            data->dev_grid_velocity_next[i * 3 + 0] = max(0., data->dev_grid_velocity_next[i * 3 + 0]);
        if (x >= data->dev_grid_size[0] - 1 - data->m_boundary_thickness)
            data->dev_grid_velocity_next[i * 3 + 0] = min(0., data->dev_grid_velocity_next[i * 3 + 0]);
        if (y <= data->m_boundary_thickness)
            data->dev_grid_velocity_next[i * 3 + 1] = max(0., data->dev_grid_velocity_next[i * 3 + 1]);
        if (y >= data->dev_grid_size[1] - 1 - data->m_boundary_thickness)
            data->dev_grid_velocity_next[i * 3 + 1] = min(0., data->dev_grid_velocity_next[i * 3 + 1]);
        if (z <= data->m_boundary_thickness)
            data->dev_grid_velocity_next[i * 3 + 2] = max(0., data->dev_grid_velocity_next[i * 3 + 2]);
        if (z >= data->dev_grid_size[2] - 1 - data->m_boundary_thickness)
            data->dev_grid_velocity_next[i * 3 + 2] = min(0., data->dev_grid_velocity_next[i * 3 + 2]);
    }

    template <typename Real>
    __global__ void Chebyshev(ImplicitMPMSolverData<Real> *data, Real omega)
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
    __global__ void G2P_velocity_and_C(ImplicitMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        if (data->dev_particle_type[i] == MPM_STATIC)
            return;
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
    __global__ void update_particle_positions(ImplicitMPMSolverData<Real> *data)
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
    __global__ void particles_boundary_conditions(ImplicitMPMSolverData<Real> *data)
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
    __global__ void swap_answer(ImplicitMPMSolverData<Real> *data)
    {
        Real *temp = data->dev_grid_velocity;
        data->dev_grid_velocity = data->dev_grid_velocity_prev;
        data->dev_grid_velocity_prev = temp;
        temp = data->dev_grid_velocity;
        data->dev_grid_velocity = data->dev_grid_velocity_next;
        data->dev_grid_velocity_next = temp;
    }
}

template <typename Real>
ImplicitMPMSolver<Real>::ImplicitMPMSolver(
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
    cudaMalloc(&m_data.dev_particle_temp_J, sizeof(Real) * m_data.m_num_particle * 9);
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
    cudaMemset(m_data.dev_grid_force, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_velocity, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_velocity, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_velocity_hat, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_velocity_hat, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_velocity_prev, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_velocity_prev, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_velocity_delta, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_velocity_delta, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_velocity_next, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_velocity_next, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_diag_K, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_diag_K, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_diag_B, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_diag_B, 0, sizeof(Real) * m_data.m_num_grid * 3);

    cudaMallocManaged(&m_data.u_energy, sizeof(Real));
    cudaMallocManaged(&m_data.u_residual, sizeof(Real));

    m_data.m_time_step = TIME_STEP;
    m_data.m_under_relaxation = UNDER_RELAXATION;
    m_data.m_ground_stiffness = GROUND_COLLISION_STIFFNESS;
    m_data.m_lame_mu = LAME_MU;
    m_data.m_lame_lambda = LAME_LAMBDA;
    cudaMalloc(&m_data.dev_gravity, sizeof(Real) * 3);
    std::vector<Real> gravity = {0., -GRAVITY, 0.};
    cudaMemcpy(m_data.dev_gravity, gravity.data(), sizeof(Real) * 3, cudaMemcpyHostToDevice);

    cudaMalloc(&m_dev_data, sizeof(ImplicitMPMSolverData<Real>));
    cudaMemcpy(m_dev_data, &m_data, sizeof(ImplicitMPMSolverData<Real>), cudaMemcpyHostToDevice);
}

template <typename Real>
ImplicitMPMSolver<Real>::~ImplicitMPMSolver()
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
    cudaFree(m_data.dev_grid_force);
    cudaFree(m_data.dev_grid_velocity);
    cudaFree(m_data.dev_grid_velocity_hat);
    cudaFree(m_data.dev_grid_velocity_prev);
    cudaFree(m_data.dev_grid_velocity_delta);
    cudaFree(m_data.dev_grid_velocity_next);
    cudaFree(m_data.dev_grid_diag_K);
    cudaFree(m_data.dev_grid_diag_B);

    cudaFree(m_data.u_energy);
    cudaFree(m_data.u_residual);

    cudaFree(m_data.dev_gravity);

    cudaFree(m_dev_data);
}

template <typename Real>
void ImplicitMPMSolver<Real>::Step()
{
    // P2G
    cudaMemset(m_data.dev_grid_momentum, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_mass, 0, sizeof(Real) * m_data.m_num_grid);
    cudaMemset(m_data.dev_grid_diag_K, 0, sizeof(Real) * m_data.m_num_grid * 3);
    ImplicitMPMSolverKernel::update_F<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    ImplicitMPMSolverKernel::particles_gravity<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    ImplicitMPMSolverKernel::calc_particle_to_leftbottom_grid_id<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    ImplicitMPMSolverKernel::P2G_momentum_and_mass_and_K<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    ImplicitMPMSolverKernel::calc_grids_velocity<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);

    switch (m_solver_type)
    {
    case NEWTON:
        NewtonSolver();
        break;
    case NEWTON_WITH_LINE_SEARCH:
        NewtonWithLineSearch();
        break;
    case CHEBYSHEV:
        ChebyshevSolver();
        break;
    }

    cudaMemset(m_data.dev_particle_velocity, 0, sizeof(Real) * m_data.m_num_particle * 3);
    cudaMemset(m_data.dev_particle_C, 0, sizeof(Real) * m_data.m_num_particle * 9);
    ImplicitMPMSolverKernel::G2P_velocity_and_C<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    ImplicitMPMSolverKernel::update_particle_positions<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    // ImplicitMPMSolverKernel::particles_boundary_conditions<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
}

template <typename Real>
Real *ImplicitMPMSolver<Real>::GetDevicePositions()
{
    return m_data.dev_particle_position;
}

template <typename Real>
void ImplicitMPMSolver<Real>::NewtonSolver()
{
    cudaMemcpy(m_data.dev_grid_velocity_hat, m_data.dev_grid_velocity, sizeof(Real) * m_data.m_num_grid * 3, cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_data.dev_grid_velocity_prev, m_data.dev_grid_velocity, sizeof(Real) * m_data.m_num_grid * 3, cudaMemcpyDeviceToDevice);
    for (unsigned int iter = 0; iter < MAX_ITERATIONS; ++iter)
    {
        cudaMemset(m_data.dev_grid_force, 0, sizeof(Real) * m_data.m_num_grid * 3);
        ImplicitMPMSolverKernel::G2P_calc_particle_temp_F<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
        ImplicitMPMSolverKernel::P2G_calc_grid_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
        ImplicitMPMSolverKernel::grid_op<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
        ImplicitMPMSolverKernel::jacobi_iteration<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
        cudaPhysics::array_axpby(m_data.dev_grid_velocity_next, (Real)1, m_data.dev_grid_velocity, (Real)1, m_data.dev_grid_velocity_delta, 3 * m_data.m_num_grid);
        ImplicitMPMSolverKernel::grids_boundary_conditions<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
        SwapAnswerBuffers();
    }
}

template <typename Real>
void ImplicitMPMSolver<Real>::NewtonWithLineSearch()
{
    cudaMemcpy(m_data.dev_grid_velocity_hat, m_data.dev_grid_velocity, sizeof(Real) * m_data.m_num_grid * 3, cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_data.dev_grid_velocity_prev, m_data.dev_grid_velocity, sizeof(Real) * m_data.m_num_grid * 3, cudaMemcpyDeviceToDevice);
    for (unsigned int iter = 0; iter < MAX_ITERATIONS; ++iter)
    {
        if (m_verbose)
            printf("Iteration %d ", iter);

        cudaMemset(m_data.dev_grid_force, 0, sizeof(Real) * m_data.m_num_grid * 3);
        ImplicitMPMSolverKernel::G2P_calc_particle_temp_F<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
        ImplicitMPMSolverKernel::P2G_calc_grid_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
        ImplicitMPMSolverKernel::grid_op<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
        cudaDeviceSynchronize();
        *m_data.u_residual = 0;
        ImplicitMPMSolverKernel::jacobi_iteration<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
        cudaDeviceSynchronize();
        Real residual = *m_data.u_residual / m_data.m_num_grid;
        if (m_verbose)
            printf("residual: %f\n", residual);
        if (residual < RESIDUAL_TOLERANCE)
            break;

        // line search
        if (iter % LINE_SEARCH_ITER == 0)
        {
            cudaMemcpy(m_data.dev_grid_velocity_next, m_data.dev_grid_velocity, sizeof(Real) * m_data.m_num_grid * 3, cudaMemcpyDeviceToDevice);
            ImplicitMPMSolverKernel::calc_particle_internal_energy<<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
            ImplicitMPMSolverKernel::calc_grid_op_energy<<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
            cudaDeviceSynchronize();
            Real initial_energy = *m_data.u_energy;
            if (m_verbose)
                printf("Step initial_energy: %f\n", initial_energy);
            Real alpha = INITIAL_ALPHA;
            while (alpha > MIN_ALPHA)
            {
                *m_data.u_energy = 0;
                cudaPhysics::array_axpby(m_data.dev_grid_velocity_next, (Real)1, m_data.dev_grid_velocity, alpha, m_data.dev_grid_velocity_delta, 3 * m_data.m_num_grid);
                ImplicitMPMSolverKernel::G2P_calc_particle_temp_F<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data, alpha);
                ImplicitMPMSolverKernel::calc_particle_internal_energy<<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
                ImplicitMPMSolverKernel::calc_grid_op_energy<<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
                cudaDeviceSynchronize();
                if (m_verbose)
                    printf("Try alpha: %f Energy: %f\n", alpha, *m_data.u_energy);
                if (*m_data.u_energy < initial_energy)
                    break;
                alpha *= ALPHA_DECAY;
            }
            *m_data.u_energy = 0;
        }
        SwapAnswerBuffers();
    }
    cudaMemcpy(m_data.dev_grid_velocity_next, m_data.dev_grid_velocity, sizeof(Real) * m_data.m_num_grid * 3, cudaMemcpyDeviceToDevice);
    ImplicitMPMSolverKernel::grids_boundary_conditions<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
    SwapAnswerBuffers();
}

template <typename Real>
void ImplicitMPMSolver<Real>::ChebyshevSolver()
{
    cudaMemcpy(m_data.dev_grid_velocity_hat, m_data.dev_grid_velocity, sizeof(Real) * m_data.m_num_grid * 3, cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_data.dev_grid_velocity_prev, m_data.dev_grid_velocity, sizeof(Real) * m_data.m_num_grid * 3, cudaMemcpyDeviceToDevice);
    Real omega = 1;
    for (unsigned int iter = 0; iter < MAX_ITERATIONS; ++iter)
    {
        cudaMemset(m_data.dev_grid_force, 0, sizeof(Real) * m_data.m_num_grid * 3);
        ImplicitMPMSolverKernel::G2P_calc_particle_temp_F<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
        ImplicitMPMSolverKernel::P2G_calc_grid_force<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
        ImplicitMPMSolverKernel::grid_op<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
        if (m_verbose)
        {
            cudaDeviceSynchronize();
            *m_data.u_residual = 0;
        }
        ImplicitMPMSolverKernel::jacobi_iteration<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
        cudaPhysics::array_axpby(m_data.dev_grid_velocity_next, (Real)1, m_data.dev_grid_velocity, (Real)1, m_data.dev_grid_velocity_delta, 3 * m_data.m_num_grid);
        ImplicitMPMSolverKernel::grids_boundary_conditions<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
        UpdateChebyshevOmega(omega, iter);
        ImplicitMPMSolverKernel::Chebyshev<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data, omega);
        SwapAnswerBuffers();
    }
    if (m_verbose)
    {
        cudaDeviceSynchronize();
        printf("Final residual: %f\n", *m_data.u_residual / m_data.m_num_grid);
    }
}

template <typename Real>
void ImplicitMPMSolver<Real>::UpdateChebyshevOmega(Real &omega, unsigned int iter)
{
    if (iter < CHEBYSHEV_DELAY_ITER)
        omega = 1;
    else if (iter == CHEBYSHEV_DELAY_ITER)
        omega = 2 / (2 - CHEBYSHEV_RHO * CHEBYSHEV_RHO);
    else
        omega = 4 / (4 - CHEBYSHEV_RHO * CHEBYSHEV_RHO * omega);
}

template <typename Real>
void ImplicitMPMSolver<Real>::SwapAnswerBuffers()
{
    std::swap(m_data.dev_grid_velocity, m_data.dev_grid_velocity_prev);
    std::swap(m_data.dev_grid_velocity, m_data.dev_grid_velocity_next);
    ImplicitMPMSolverKernel::swap_answer<Real><<<1, 1>>>(m_dev_data);
}

template class ImplicitMPMSolver<float>;
template class ImplicitMPMSolver<double>;
template struct ImplicitMPMSolverData<float>;
template struct ImplicitMPMSolverData<double>;

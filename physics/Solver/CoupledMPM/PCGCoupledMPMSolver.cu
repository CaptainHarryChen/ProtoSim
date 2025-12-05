#include "PCGCoupledMPMSolver.cuh"
#include <cuda_utils/cuda_utils.cuh>
#include <cuda_utils/error.cuh>
#include <cuda_utils/array.cuh>
#include <Math/algebra.cuh>
#include <Math/geometry.cuh>
#include <Math/ConstitutiveModel/CorotatedLinear.cuh>
#include <Math/ConstitutiveModel/Neohookean.cuh>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include <cassert>
#include <cmath>
#include <cstdio>

// Toggle constitutive model for the FEM solve here.
// #define COUPLED_FEM_NEWHOOKEAN_MODEL
#define COUPLED_FEM_COROTATED_LINEAR_MODEL

#if (defined(COUPLED_FEM_NEWHOOKEAN_MODEL) && defined(COUPLED_FEM_COROTATED_LINEAR_MODEL)) || (!defined(COUPLED_FEM_NEWHOOKEAN_MODEL) && !defined(COUPLED_FEM_COROTATED_LINEAR_MODEL))
#error "Select exactly one constitutive model for the coupled FEM stage."
#endif

namespace CoupledFemKernel
{
    template <typename Real>
    __global__ void initialize_tetrahedra(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= data->fem_tetrahedron_count)
            return;

        unsigned int idx[4];
        for (unsigned int i = 0; i < 4; ++i)
            idx[i] = data->dev_fem_tetrahedron[t * 4 + i] * 3;

        Real Dm[9];
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                Dm[i * 3 + j] = data->dev_fem_node_position[idx[j + 1] + i] - data->dev_fem_node_position[idx[0] + i];

        Real vol = cudaPhysics::det3(Dm);
        data->dev_fem_tet_volume[t] = fabs(vol) / (Real)6.0;
        cudaPhysics::matInv3(&data->dev_fem_inv_dm[t * 9], Dm);

        for (unsigned int i = 0; i < 4; ++i)
        {
            unsigned int node = data->dev_fem_tetrahedron[t * 4 + i];
            atomicAdd(&data->dev_fem_node_mass[node], (Real)0.25 * data->dev_fem_tet_volume[t] * data->dev_fem_tet_density[t]);
        }
    }

    template <typename Real>
    __global__ void predict_node_states(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->fem_node_count)
            return;

        cudaPhysics::axpby(&data->dev_fem_node_velocity[3 * v], (Real)1.0, &data->dev_fem_node_velocity[3 * v], data->fem_time_step, data->dev_gravity, 3);
        cudaPhysics::axpby(&data->dev_fem_node_position_next[3 * v], (Real)1.0, &data->dev_fem_node_position[3 * v], data->fem_time_step, &data->dev_fem_node_velocity[3 * v], 3);
    }

    template <typename Real>
    __global__ void assemble_tet_diagonal(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= data->fem_tetrahedron_count)
            return;

#ifdef COUPLED_FEM_NEWHOOKEAN_MODEL
        unsigned int idx[4];
        for (unsigned int i = 0; i < 4; ++i)
            idx[i] = data->dev_fem_tetrahedron[t * 4 + i] * 3;
        Real K[12];
        cudaPhysics::calc_neohookean_K_diag<Real>(
            K,
            &data->dev_fem_node_position_next[idx[0]],
            &data->dev_fem_node_position_next[idx[1]],
            &data->dev_fem_node_position_next[idx[2]],
            &data->dev_fem_node_position_next[idx[3]],
            &data->dev_fem_inv_dm[t * 9],
            data->dev_fem_tet_volume[t],
            data->fem_lame_mu,
            data->fem_lame_lambda);
        for (unsigned int i = 0; i < 4; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                atomicAdd(&data->dev_fem_node_diag[idx[i] + j], K[i * 3 + j]);
#endif

#ifdef COUPLED_FEM_COROTATED_LINEAR_MODEL
        Real K[4];
        cudaPhysics::calc_corotated_linear_K_diag<Real>(
            K,
            &data->dev_fem_inv_dm[t * 9],
            data->dev_fem_tet_volume[t],
            data->fem_lame_mu);
        for (unsigned int i = 0; i < 4; ++i)
        {
            unsigned int node = data->dev_fem_tetrahedron[t * 4 + i];
            for (unsigned int j = 0; j < 3; ++j)
                atomicAdd(&data->dev_fem_node_diag[node * 3 + j], K[i]);
        }
#endif
    }

    template <typename Real>
    __global__ void finalize_diagonal(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->fem_node_count)
            return;

        for (unsigned int j = 0; j < 3; ++j)
        {
            Real mass_term = data->dev_fem_node_mass[v] * data->fem_time_step_inv;
            data->dev_fem_node_diag[v * 3 + j] = mass_term - data->dev_fem_node_diag[v * 3 + j] * data->fem_time_step;
        }
    }

    template <typename Real>
    __global__ void accumulate_tet_force(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= data->fem_tetrahedron_count)
            return;

        unsigned int idx[4];
        for (unsigned int i = 0; i < 4; ++i)
            idx[i] = data->dev_fem_tetrahedron[t * 4 + i] * 3;

        Real Ds[9];
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                Ds[i * 3 + j] = data->dev_fem_node_position_next[idx[j + 1] + i] - data->dev_fem_node_position_next[idx[0] + i];

        const Real *InvDm = &data->dev_fem_inv_dm[t * 9];
        Real F[9], P[9], H[9], f0[3] = {0};
        cudaPhysics::matMul3(F, Ds, InvDm);
#ifdef COUPLED_FEM_NEWHOOKEAN_MODEL
        cudaPhysics::calc_neohookean_P<Real>(P, F, data->fem_lame_mu, data->fem_lame_lambda);
#endif
#ifdef COUPLED_FEM_COROTATED_LINEAR_MODEL
        cudaPhysics::calc_corotated_linear_P<Real>(P, F, data->fem_lame_mu);
#endif
        cudaPhysics::matmatTMul3(H, P, InvDm);
        cudaPhysics::vecMul(H, -data->dev_fem_tet_volume[t], H, 9);
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                f0[i] -= H[i * 3 + j];

        for (unsigned int j = 0; j < 3; ++j)
            atomicAdd(&data->dev_fem_node_force[idx[0] + j], f0[j]);
        for (unsigned int i = 1; i < 4; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                atomicAdd(&data->dev_fem_node_force[idx[i] + j], H[j * 3 + i - 1]);
    }

    template <typename Real>
    __global__ void apply_ground_contact(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->fem_node_count)
            return;

        if (data->dev_fem_node_position_next[3 * v + 1] < 0.0f)
        {
            Real penalty = -data->fem_collision_stiffness * data->dev_fem_node_position_next[3 * v + 1] * data->dev_fem_node_mass[v];
            data->dev_fem_node_force[3 * v + 1] += penalty;
            Real diag_update = data->fem_collision_stiffness * data->dev_fem_node_mass[v] * data->fem_time_step;
            data->dev_fem_node_diag[3 * v + 1] += diag_update;
            data->dev_fem_node_collision_diag[3 * v + 1] += diag_update;
        }
    }

    template <typename Real>
    __global__ void residual_norm(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->fem_node_count)
            return;

        for (unsigned int j = 0; j < 3; ++j)
        {
            Real r = -(data->dev_fem_node_mass[v] * data->fem_time_step_inv *
                           (data->dev_fem_node_velocity[3 * v + j] - data->dev_fem_node_velocity_hat[3 * v + j]) -
                       data->dev_fem_node_force[3 * v + j]);
            data->dev_fem_scratch[3 * v + j] = r * r;
        }
    }

    template <typename Real>
    __global__ void preconditioned_residual(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->fem_node_count)
            return;

        for (unsigned int j = 0; j < 3; ++j)
        {
            Real r = -(data->dev_fem_node_mass[v] * data->fem_time_step_inv *
                           (data->dev_fem_node_velocity[3 * v + j] - data->dev_fem_node_velocity_hat[3 * v + j]) -
                       data->dev_fem_node_force[3 * v + j]);
            data->dev_fem_scratch[3 * v + j] = r * r / data->dev_fem_node_diag[3 * v + j];
        }
    }

    template <typename Real>
    __global__ void search_direction(const PCGCoupledMPMSolverData<Real> *data, Real beta)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->fem_node_count)
            return;

        for (unsigned int j = 0; j < 3; ++j)
        {
            Real r = -(data->dev_fem_node_mass[v] * data->fem_time_step_inv *
                           (data->dev_fem_node_velocity[3 * v + j] - data->dev_fem_node_velocity_hat[3 * v + j]) -
                       data->dev_fem_node_force[3 * v + j]);
            data->dev_fem_search_direction[3 * v + j] = r / data->dev_fem_node_diag[3 * v + j] +
                                                        beta * data->dev_fem_search_direction[3 * v + j];
        }
    }

    template <typename Real>
    __global__ void accumulate_tet_ap(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= data->fem_tetrahedron_count)
            return;

        unsigned int idx[4];
        for (unsigned int i = 0; i < 4; ++i)
            idx[i] = data->dev_fem_tetrahedron[t * 4 + i] * 3;
        const Real *InvDm = &data->dev_fem_inv_dm[t * 9];
        Real partial_Ds[9];
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                partial_Ds[i * 3 + j] = (data->dev_fem_search_direction[idx[j + 1] + i] - data->dev_fem_search_direction[idx[0] + i]) * data->fem_time_step;

        Real Ku[12];
#ifdef COUPLED_FEM_NEWHOOKEAN_MODEL
        Real Ds[9];
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                Ds[i * 3 + j] = data->dev_fem_node_position_next[idx[j + 1] + i] - data->dev_fem_node_position_next[idx[0] + i];
        cudaPhysics::calc_neohookean_K_mul_u<Real>(
            Ku,
            Ds,
            partial_Ds,
            InvDm,
            data->dev_fem_tet_volume[t],
            data->fem_lame_mu,
            data->fem_lame_lambda);
#endif
#ifdef COUPLED_FEM_COROTATED_LINEAR_MODEL
        cudaPhysics::calc_corotated_linear_K_mul_u<Real>(
            Ku,
            partial_Ds,
            InvDm,
            data->dev_fem_tet_volume[t],
            data->fem_lame_mu);
#endif

        for (unsigned int i = 0; i < 4; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                atomicAdd(&data->dev_fem_scratch[idx[i] + j], Ku[i * 3 + j]);
    }

    template <typename Real>
    __global__ void finalize_node_pap(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->fem_node_count)
            return;

        for (unsigned int j = 0; j < 3; ++j)
        {
            Real Ap = data->dev_fem_node_mass[v] * data->fem_time_step_inv * data->dev_fem_search_direction[3 * v + j] -
                      data->dev_fem_scratch[3 * v + j] +
                      data->dev_fem_node_collision_diag[3 * v + j] * data->dev_fem_search_direction[3 * v + j];
            data->dev_fem_scratch[3 * v + j] = data->dev_fem_search_direction[3 * v + j] * Ap;
        }
    }

    template <typename Real>
    __global__ void search_dot_residual(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->fem_node_count)
            return;

        for (unsigned int j = 0; j < 3; ++j)
        {
            Real r = -(data->dev_fem_node_mass[v] * data->fem_time_step_inv *
                           (data->dev_fem_node_velocity[3 * v + j] - data->dev_fem_node_velocity_hat[3 * v + j]) -
                       data->dev_fem_node_force[3 * v + j]);
            data->dev_fem_scratch[3 * v + j] = data->dev_fem_search_direction[3 * v + j] * r;
        }
    }
}

namespace CoupledSharedKernel
{
    template <typename Real>
    __global__ void update_sample_positions(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int s = blockDim.x * blockIdx.x + threadIdx.x;
        if (s >= data->sample_count)
            return;

        unsigned int tri = data->dev_sample_triangle_index[s];
        unsigned int node_idx[3];
        for (unsigned int i = 0; i < 3; ++i)
            node_idx[i] = data->dev_surface_triangles[tri * 3 + i];

        const Real *bary = &data->dev_sample_barycentric[s * 3];
        Real *dst = &data->dev_sample_position[s * 3];
        dst[0] = dst[1] = dst[2] = (Real)0;

        for (unsigned int i = 0; i < 3; ++i)
        {
            const Real *node_pos = &data->dev_fem_node_position[node_idx[i] * 3];
            dst[0] += bary[i] * node_pos[0];
            dst[1] += bary[i] * node_pos[1];
            dst[2] += bary[i] * node_pos[2];
        }
    }
}

namespace CoupledMpmKernel
{
    template <typename Real>
    __global__ void update_particle_F(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->mpm_particle_count)
            return;
        assert(data->dev_mpm_particle_type[i] == MPM_FLUID);
        Real trace_C = data->dev_mpm_particle_C[i * 9] + data->dev_mpm_particle_C[i * 9 + 4] + data->dev_mpm_particle_C[i * 9 + 8];
        Real J = data->dev_mpm_particle_F[i * 9] * (1 + data->mpm_time_step * trace_C);
        data->dev_mpm_particle_F[i * 9] = J;
    }

    template <typename Real>
    __global__ void assign_particle_cells(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->mpm_particle_count)
            return;

        int x = static_cast<int>(floor((data->dev_mpm_particle_position[i * 3 + 0] - data->dev_mpm_outer_bbox[0]) / data->mpm_grid_spacing + 0.5f)) - 1;
        int y = static_cast<int>(floor((data->dev_mpm_particle_position[i * 3 + 1] - data->dev_mpm_outer_bbox[1]) / data->mpm_grid_spacing + 0.5f)) - 1;
        int z = static_cast<int>(floor((data->dev_mpm_particle_position[i * 3 + 2] - data->dev_mpm_outer_bbox[2]) / data->mpm_grid_spacing + 0.5f)) - 1;

        int max_x = static_cast<int>(data->dev_mpm_grid_resolution[0]) - 3;
        int max_y = static_cast<int>(data->dev_mpm_grid_resolution[1]) - 3;
        int max_z = static_cast<int>(data->dev_mpm_grid_resolution[2]) - 3;

        if (x < 0 || y < 0 || z < 0 || x > max_x || y > max_y || z > max_z)
        {
            x = max(0, min(x, max_x));
            y = max(0, min(y, max_y));
            z = max(0, min(z, max_z));
            printf("Warning: particle %u is out of grid, set to %d %d %d [Particle position] %.10f %.10f %.10f\n",
                   i, x, y, z,
                   data->dev_mpm_particle_position[i * 3 + 0],
                   data->dev_mpm_particle_position[i * 3 + 1],
                   data->dev_mpm_particle_position[i * 3 + 2]);
        }

        data->dev_mpm_particle_cell[i] = x * data->dev_mpm_grid_resolution[1] * data->dev_mpm_grid_resolution[2] +
                                         y * data->dev_mpm_grid_resolution[2] + z;
    }

    template <typename Real>
    __device__ void grid_position(Real *grid_position, unsigned int id, const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int z = id % data->dev_mpm_grid_resolution[2];
        unsigned int y = (id / data->dev_mpm_grid_resolution[2]) % data->dev_mpm_grid_resolution[1];
        unsigned int x = id / data->dev_mpm_grid_resolution[2] / data->dev_mpm_grid_resolution[1];

        grid_position[0] = data->dev_mpm_outer_bbox[0] + x * data->mpm_grid_spacing;
        grid_position[1] = data->dev_mpm_outer_bbox[1] + y * data->mpm_grid_spacing;
        grid_position[2] = data->dev_mpm_outer_bbox[2] + z * data->mpm_grid_spacing;
    }

    template <typename Real>
    __device__ Real quadratic_weight(const Real *grid_position, const Real *particle_position, Real grid_spacing)
    {
        Real result = 1;
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
    __global__ void p2g_momentum_mass_diag(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int p = blockDim.x * blockIdx.x + threadIdx.x;
        if (p >= data->mpm_particle_count)
            return;

        Real coef = -data->mpm_fluid_lambda *
                    data->mpm_time_step *
                    (Real)16 / (data->mpm_grid_spacing * data->mpm_grid_spacing * data->mpm_grid_spacing * data->mpm_grid_spacing) *
                    data->dev_mpm_particle_volume[p] *
                    (data->dev_mpm_particle_F[p * 9] * data->dev_mpm_particle_F[p * 9]);
        coef += -data->mpm_fluid_viscosity *
                (Real)64 / (3 * data->mpm_grid_spacing * data->mpm_grid_spacing * data->mpm_grid_spacing * data->mpm_grid_spacing) *
                data->dev_mpm_particle_volume[p] * data->dev_mpm_particle_F[p * 9];

        const Real *particle_position = &data->dev_mpm_particle_position[p * 3];
        const Real *particle_velocity = &data->dev_mpm_particle_velocity[p * 3];
        unsigned int leftbottom = data->dev_mpm_particle_cell[p];
        unsigned int x = leftbottom / data->dev_mpm_grid_resolution[1] / data->dev_mpm_grid_resolution[2];
        unsigned int y = (leftbottom / data->dev_mpm_grid_resolution[2]) % data->dev_mpm_grid_resolution[1];
        unsigned int z = leftbottom % data->dev_mpm_grid_resolution[2];

        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data->dev_mpm_grid_resolution[1] * data->dev_mpm_grid_resolution[2] +
                                           (y + dy) * data->dev_mpm_grid_resolution[2] + (z + dz);
                    Real grid_pos[3];
                    grid_position(grid_pos, grid_id, data);
                    Real weight = quadratic_weight(grid_pos, particle_position, data->mpm_grid_spacing);

                    Real momentum[3];
                    Real delta[3];
                    cudaPhysics::vecSubs3(delta, grid_pos, particle_position);
                    cudaPhysics::matVec3(momentum, &data->dev_mpm_particle_C[p * 9], delta);
                    cudaPhysics::vecMul3(momentum, data->dev_mpm_particle_mass[p], momentum);

                    Real temp[3];
                    cudaPhysics::vecMul3(temp, data->dev_mpm_particle_mass[p], particle_velocity);
                    cudaPhysics::vecAdd3(momentum, temp, momentum);
                    cudaPhysics::vecMul3(momentum, weight, momentum);

                    atomicAdd(&data->dev_mpm_grid_momentum[grid_id * 3 + 0], momentum[0]);
                    atomicAdd(&data->dev_mpm_grid_momentum[grid_id * 3 + 1], momentum[1]);
                    atomicAdd(&data->dev_mpm_grid_momentum[grid_id * 3 + 2], momentum[2]);

                    atomicAdd(&data->dev_mpm_grid_mass[grid_id], weight * data->dev_mpm_particle_mass[p]);

                    Real diagK[3];
                    cudaPhysics::vecMul3(diagK, delta, delta);
                    cudaPhysics::vecMul3(diagK, coef * weight * weight, diagK);
                    atomicAdd(&data->dev_mpm_grid_diag_const[grid_id * 3 + 0], diagK[0]);
                    atomicAdd(&data->dev_mpm_grid_diag_const[grid_id * 3 + 1], diagK[1]);
                    atomicAdd(&data->dev_mpm_grid_diag_const[grid_id * 3 + 2], diagK[2]);
                }
    }

    template <typename Real>
    __global__ void prepare_grid_values(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data->mpm_grid_count)
            return;

        if (data->dev_mpm_grid_mass[i] > 0)
        {
            cudaPhysics::vecMul3(&data->dev_mpm_grid_velocity[i * 3], (Real)(1.0 / data->dev_mpm_grid_mass[i]), &data->dev_mpm_grid_momentum[i * 3]);
            for (unsigned int j = 0; j < 3; ++j)
                data->dev_mpm_grid_diag_const[i * 3 + j] = 1 - data->mpm_time_step / data->dev_mpm_grid_mass[i] * data->dev_mpm_grid_diag_const[i * 3 + j];
        }
        else
        {
            data->dev_mpm_grid_velocity[i * 3 + 0] = 0;
            data->dev_mpm_grid_velocity[i * 3 + 1] = 0;
            data->dev_mpm_grid_velocity[i * 3 + 2] = 0;
        }
    }

    template <typename Real>
    __global__ void particles_gravity(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data->mpm_particle_count)
            return;

        Real delta[3];
        cudaPhysics::vecMul3(delta, data->mpm_time_step, data->dev_gravity);
        cudaPhysics::vecAdd3(&data->dev_mpm_particle_velocity[i * 3], &data->dev_mpm_particle_velocity[i * 3], delta);
    }

    template <typename Real>
    __global__ void compute_particle_temp_C(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data->mpm_particle_count)
            return;

        Real accum[9] = {0};
        const Real *particle_pos = &data->dev_mpm_particle_position[i * 3];
        unsigned int leftbottom = data->dev_mpm_particle_cell[i];
        unsigned int x = leftbottom / data->dev_mpm_grid_resolution[1] / data->dev_mpm_grid_resolution[2];
        unsigned int y = (leftbottom / data->dev_mpm_grid_resolution[2]) % data->dev_mpm_grid_resolution[1];
        unsigned int z = leftbottom % data->dev_mpm_grid_resolution[2];

        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data->dev_mpm_grid_resolution[1] * data->dev_mpm_grid_resolution[2] +
                                           (y + dy) * data->dev_mpm_grid_resolution[2] + (z + dz);
                    Real grid_pos[3];
                    grid_position(grid_pos, grid_id, data);
                    Real weight = quadratic_weight(grid_pos, particle_pos, data->mpm_grid_spacing);

                    Real delta[3], temp[9];
                    cudaPhysics::vecSubs3(delta, grid_pos, particle_pos);
                    cudaPhysics::vecvecT(temp, &data->dev_mpm_grid_velocity[grid_id * 3], delta, 3, 3);
                    cudaPhysics::axpby(accum, (Real)1, accum, weight, temp, 9);
                }

        cudaPhysics::vecMul(&data->dev_mpm_particle_temp_C[i * 9], (Real)4 / (data->mpm_grid_spacing * data->mpm_grid_spacing), accum, 9);
    }

    template <typename Real>
    __global__ void p2g_grid_force(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data->mpm_particle_count)
            return;

        const Real *particle_pos = &data->dev_mpm_particle_position[i * 3];
        Real tr_C = data->dev_mpm_particle_temp_C[i * 9] + data->dev_mpm_particle_temp_C[i * 9 + 4] + data->dev_mpm_particle_temp_C[i * 9 + 8];
        Real J_new = data->dev_mpm_particle_F[i * 9] * (1 + data->mpm_time_step * tr_C);
        Real elastic_coef = -data->dev_mpm_particle_volume[i] *
                            data->mpm_fluid_lambda * (J_new - 1) *
                            data->dev_mpm_particle_F[i * 9] *
                            (Real)4 / (data->mpm_grid_spacing * data->mpm_grid_spacing);

        Real viscosity_coef[9] = {0};
        viscosity_coef[0] = viscosity_coef[4] = viscosity_coef[8] = -(Real)0.66666667 * tr_C;
        cudaPhysics::axpby(viscosity_coef, (Real)1, viscosity_coef, (Real)2, &data->dev_mpm_particle_temp_C[i * 9], 9);
        cudaPhysics::vecMul(viscosity_coef,
                            -data->mpm_fluid_viscosity * data->dev_mpm_particle_volume[i] * data->dev_mpm_particle_F[i * 9] * (Real)4 /
                                (data->mpm_grid_spacing * data->mpm_grid_spacing),
                            viscosity_coef,
                            9);

        unsigned int leftbottom = data->dev_mpm_particle_cell[i];
        unsigned int x = leftbottom / data->dev_mpm_grid_resolution[1] / data->dev_mpm_grid_resolution[2];
        unsigned int y = (leftbottom / data->dev_mpm_grid_resolution[2]) % data->dev_mpm_grid_resolution[1];
        unsigned int z = leftbottom % data->dev_mpm_grid_resolution[2];

        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data->dev_mpm_grid_resolution[1] * data->dev_mpm_grid_resolution[2] +
                                           (y + dy) * data->dev_mpm_grid_resolution[2] + (z + dz);
                    Real grid_pos[3];
                    grid_position(grid_pos, grid_id, data);
                    Real weight = quadratic_weight(grid_pos, particle_pos, data->mpm_grid_spacing);

                    Real delta[3];
                    cudaPhysics::vecSubs3(delta, grid_pos, particle_pos);
                    Real elastic_force[3];
                    cudaPhysics::vecMul3(elastic_force, elastic_coef * weight, delta);

                    Real viscosity_force[3];
                    cudaPhysics::matVec3(viscosity_force, viscosity_coef, delta);
                    cudaPhysics::vecMul3(viscosity_force, weight, viscosity_force);

                    atomicAdd(&data->dev_mpm_grid_force[grid_id * 3 + 0], elastic_force[0] + viscosity_force[0]);
                    atomicAdd(&data->dev_mpm_grid_force[grid_id * 3 + 1], elastic_force[1] + viscosity_force[1]);
                    atomicAdd(&data->dev_mpm_grid_force[grid_id * 3 + 2], elastic_force[2] + viscosity_force[2]);
                }
    }

    template <typename Real>
    __global__ void grid_boundary_force(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data->mpm_grid_count)
            return;

        Real grid_pos[3];
        grid_position(grid_pos, i, data);
        cudaPhysics::axpby(grid_pos, (Real)1, grid_pos, data->mpm_time_step, &data->dev_mpm_grid_velocity[i * 3], 3);

        Real *diag_B = &data->dev_mpm_grid_diag_mutable[i * 3];
        Real *grid_force = &data->dev_mpm_grid_force[i * 3];
        for (unsigned int j = 0; j < 3; ++j)
        {
            if (grid_pos[j] <= data->dev_mpm_inner_bbox[j])
            {
                diag_B[j] += data->mpm_ground_stiffness * data->mpm_time_step * data->mpm_time_step;
                grid_force[j] += data->dev_mpm_grid_mass[i] * data->mpm_ground_stiffness * (data->dev_mpm_inner_bbox[j] - grid_pos[j]);
            }
            if (grid_pos[j] >= data->dev_mpm_inner_bbox[j + 3])
            {
                diag_B[j] += data->mpm_ground_stiffness * data->mpm_time_step * data->mpm_time_step;
                grid_force[j] += data->dev_mpm_grid_mass[i] * data->mpm_ground_stiffness * (data->dev_mpm_inner_bbox[j + 3] - grid_pos[j]);
            }
        }
    }

    template <typename Real>
    __global__ void grid_residual_norm(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data->mpm_grid_count)
            return;

        for (unsigned int j = 0; j < 3; ++j)
        {
            Real r = data->dev_mpm_grid_velocity[i * 3 + j] - data->dev_mpm_grid_velocity_hat[i * 3 + j] -
                     data->mpm_time_step / data->dev_mpm_grid_mass[i] * data->dev_mpm_grid_force[i * 3 + j];
            data->dev_mpm_grid_temp[i * 3 + j] = (data->dev_mpm_grid_mass[i] > 0) ? r * r : 0;
        }
    }

    template <typename Real>
    __global__ void grid_preconditioned_residual(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data->mpm_grid_count)
            return;

        for (unsigned int j = 0; j < 3; ++j)
        {
            Real denom = data->dev_mpm_grid_diag_const[i * 3 + j] + data->dev_mpm_grid_diag_mutable[i * 3 + j];
            Real r = data->dev_mpm_grid_velocity[i * 3 + j] - data->dev_mpm_grid_velocity_hat[i * 3 + j] -
                     data->mpm_time_step / data->dev_mpm_grid_mass[i] * data->dev_mpm_grid_force[i * 3 + j];
            data->dev_mpm_grid_temp[i * 3 + j] = (data->dev_mpm_grid_mass[i] > 0) ? r * r / denom : 0;
        }
    }

    template <typename Real>
    __global__ void grid_search_direction(const PCGCoupledMPMSolverData<Real> *data, Real beta)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data->mpm_grid_count)
            return;

        for (unsigned int j = 0; j < 3; ++j)
        {
            Real denom = data->dev_mpm_grid_diag_const[i * 3 + j] + data->dev_mpm_grid_diag_mutable[i * 3 + j];
            Real r = data->dev_mpm_grid_velocity[i * 3 + j] - data->dev_mpm_grid_velocity_hat[i * 3 + j] -
                     data->mpm_time_step / data->dev_mpm_grid_mass[i] * data->dev_mpm_grid_force[i * 3 + j];
            if (data->dev_mpm_grid_mass[i] > 0)
                data->dev_mpm_grid_search[i * 3 + j] = -r / denom + beta * data->dev_mpm_grid_search[i * 3 + j];
            else
                data->dev_mpm_grid_search[i * 3 + j] = 0;
        }
    }

    template <typename Real>
    __global__ void g2p_calc_Ap_step1(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int p = blockDim.x * blockIdx.x + threadIdx.x;
        if (p >= data->mpm_particle_count)
            return;

        Real acc = 0;
        const Real *particle_pos = &data->dev_mpm_particle_position[p * 3];
        unsigned int leftbottom = data->dev_mpm_particle_cell[p];
        unsigned int x = leftbottom / data->dev_mpm_grid_resolution[1] / data->dev_mpm_grid_resolution[2];
        unsigned int y = (leftbottom / data->dev_mpm_grid_resolution[2]) % data->dev_mpm_grid_resolution[1];
        unsigned int z = leftbottom % data->dev_mpm_grid_resolution[2];

        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data->dev_mpm_grid_resolution[1] * data->dev_mpm_grid_resolution[2] +
                                           (y + dy) * data->dev_mpm_grid_resolution[2] + (z + dz);
                    Real grid_pos[3];
                    grid_position(grid_pos, grid_id, data);
                    Real weight = quadratic_weight(grid_pos, particle_pos, data->mpm_grid_spacing);

                    Real delta[3];
                    cudaPhysics::vecSubs3(delta, grid_pos, particle_pos);
                    acc += weight * cudaPhysics::dot3(&data->dev_mpm_grid_search[grid_id * 3], delta);
                }
        data->dev_mpm_particle_scalar[p] = acc;
    }

    template <typename Real>
    __global__ void p2g_calc_Ap_step2(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int p = blockDim.x * blockIdx.x + threadIdx.x;
        if (p >= data->mpm_particle_count)
            return;

        Real coef = -data->mpm_fluid_lambda *
                    data->mpm_time_step *
                    (Real)16 / (data->mpm_grid_spacing * data->mpm_grid_spacing * data->mpm_grid_spacing * data->mpm_grid_spacing) *
                    data->dev_mpm_particle_volume[p] *
                    (data->dev_mpm_particle_F[p * 9] * data->dev_mpm_particle_F[p * 9]) *
                    data->dev_mpm_particle_scalar[p];
        coef += -data->mpm_fluid_viscosity *
                (Real)64 / (3 * data->mpm_grid_spacing * data->mpm_grid_spacing * data->mpm_grid_spacing * data->mpm_grid_spacing) *
                data->dev_mpm_particle_volume[p] * data->dev_mpm_particle_F[p * 9] *
                data->dev_mpm_particle_scalar[p];

        const Real *particle_pos = &data->dev_mpm_particle_position[p * 3];
        unsigned int leftbottom = data->dev_mpm_particle_cell[p];
        unsigned int x = leftbottom / data->dev_mpm_grid_resolution[1] / data->dev_mpm_grid_resolution[2];
        unsigned int y = (leftbottom / data->dev_mpm_grid_resolution[2]) % data->dev_mpm_grid_resolution[1];
        unsigned int z = leftbottom % data->dev_mpm_grid_resolution[2];

        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data->dev_mpm_grid_resolution[1] * data->dev_mpm_grid_resolution[2] +
                                           (y + dy) * data->dev_mpm_grid_resolution[2] + (z + dz);
                    Real grid_pos[3];
                    grid_position(grid_pos, grid_id, data);
                    Real weight = quadratic_weight(grid_pos, particle_pos, data->mpm_grid_spacing);

                    Real delta[3];
                    cudaPhysics::vecSubs3(delta, grid_pos, particle_pos);
                    atomicAdd(&data->dev_mpm_grid_temp[grid_id * 3 + 0], coef * weight * delta[0]);
                    atomicAdd(&data->dev_mpm_grid_temp[grid_id * 3 + 1], coef * weight * delta[1]);
                    atomicAdd(&data->dev_mpm_grid_temp[grid_id * 3 + 2], coef * weight * delta[2]);
                }
    }

    template <typename Real>
    __global__ void grid_calc_pAp_step3(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data->mpm_grid_count)
            return;

        Real Ap_nondiag[3];
        cudaPhysics::vecMul3(Ap_nondiag, -data->mpm_time_step / data->dev_mpm_grid_mass[i], &data->dev_mpm_grid_temp[i * 3]);
        Real Ap_diag[3];
        cudaPhysics::vecMul3(Ap_diag, &data->dev_mpm_grid_diag_mutable[i * 3], &data->dev_mpm_grid_search[i * 3]);

        Real total[3];
        cudaPhysics::axpbypcz(total, (Real)1, &data->dev_mpm_grid_search[i * 3], (Real)1, Ap_nondiag, (Real)1, Ap_diag, 3);
        cudaPhysics::vecMul3(&data->dev_mpm_grid_temp[i * 3], &data->dev_mpm_grid_search[i * 3], total);

        if (data->dev_mpm_grid_mass[i] <= 0)
        {
            data->dev_mpm_grid_temp[i * 3 + 0] = 0;
            data->dev_mpm_grid_temp[i * 3 + 1] = 0;
            data->dev_mpm_grid_temp[i * 3 + 2] = 0;
        }
    }

    template <typename Real>
    __global__ void grid_search_dot_residual(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data->mpm_grid_count)
            return;

        for (unsigned int j = 0; j < 3; ++j)
        {
            Real r = data->dev_mpm_grid_velocity[i * 3 + j] - data->dev_mpm_grid_velocity_hat[i * 3 + j] -
                     data->mpm_time_step / data->dev_mpm_grid_mass[i] * data->dev_mpm_grid_force[i * 3 + j];
            data->dev_mpm_grid_temp[i * 3 + j] = (data->dev_mpm_grid_mass[i] > 0) ? data->dev_mpm_grid_search[i * 3 + j] * r : 0;
        }
    }

    template <typename Real>
    __global__ void enforce_grid_boundaries(PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data->mpm_grid_count)
            return;

        Real grid_pos[3];
        grid_position(grid_pos, i, data);
        for (unsigned int j = 0; j < 3; ++j)
        {
            Real predicted = grid_pos[j] + data->dev_mpm_grid_velocity[i * 3 + j] * data->mpm_time_step;
            if (predicted < data->dev_mpm_inner_bbox[j])
                data->dev_mpm_grid_velocity[i * 3 + j] = (data->dev_mpm_inner_bbox[j] - grid_pos[j]) / data->mpm_time_step;
            if (predicted > data->dev_mpm_inner_bbox[j + 3])
                data->dev_mpm_grid_velocity[i * 3 + j] = (data->dev_mpm_inner_bbox[j + 3] - grid_pos[j]) / data->mpm_time_step;
        }
    }

    template <typename Real>
    __global__ void g2p_velocity_and_C(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data->mpm_particle_count)
            return;

        const Real *particle_pos = &data->dev_mpm_particle_position[i * 3];
        unsigned int leftbottom = data->dev_mpm_particle_cell[i];
        unsigned int x = leftbottom / data->dev_mpm_grid_resolution[1] / data->dev_mpm_grid_resolution[2];
        unsigned int y = (leftbottom / data->dev_mpm_grid_resolution[2]) % data->dev_mpm_grid_resolution[1];
        unsigned int z = leftbottom % data->dev_mpm_grid_resolution[2];

        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data->dev_mpm_grid_resolution[1] * data->dev_mpm_grid_resolution[2] +
                                           (y + dy) * data->dev_mpm_grid_resolution[2] + (z + dz);
                    Real grid_pos[3];
                    grid_position(grid_pos, grid_id, data);
                    Real weight = quadratic_weight(grid_pos, particle_pos, data->mpm_grid_spacing);

                    Real velocity[3];
                    cudaPhysics::vecMul3(velocity, weight, &data->dev_mpm_grid_velocity[grid_id * 3]);
                    cudaPhysics::vecAdd3(&data->dev_mpm_particle_velocity[i * 3], &data->dev_mpm_particle_velocity[i * 3], velocity);

                    Real delta[3];
                    cudaPhysics::vecSubs3(delta, grid_pos, particle_pos);
                    Real temp_C[9];
                    cudaPhysics::vecvecT(temp_C, &data->dev_mpm_grid_velocity[grid_id * 3], delta, 3, 3);
                    cudaPhysics::matMul3(temp_C, weight * (Real)4 / (data->mpm_grid_spacing * data->mpm_grid_spacing), temp_C);
                    cudaPhysics::vecAdd(&data->dev_mpm_particle_C[i * 9], temp_C, &data->dev_mpm_particle_C[i * 9], 9);
                }
    }

    template <typename Real>
    __global__ void update_particle_positions(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data->mpm_particle_count)
            return;
        if (data->dev_mpm_particle_type[i] == MPM_STATIC)
            return;

        Real delta[3];
        cudaPhysics::vecMul3(delta, data->mpm_time_step, &data->dev_mpm_particle_velocity[i * 3]);
        cudaPhysics::vecAdd3(&data->dev_mpm_particle_position[i * 3], &data->dev_mpm_particle_position[i * 3], delta);
    }

    template <typename Real>
    __global__ void particles_boundary_conditions(const PCGCoupledMPMSolverData<Real> *data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data->mpm_particle_count)
            return;

        for (unsigned int j = 0; j < 3; ++j)
        {
            if (data->dev_mpm_particle_position[i * 3 + j] < data->dev_mpm_inner_bbox[j])
                data->dev_mpm_particle_position[i * 3 + j] = data->dev_mpm_inner_bbox[j];
            if (data->dev_mpm_particle_position[i * 3 + j] > data->dev_mpm_inner_bbox[j + 3])
                data->dev_mpm_particle_position[i * 3 + j] = data->dev_mpm_inner_bbox[j + 3];
        }
    }
}

namespace
{
    template <typename Real>
    void RunFemStage(PCGCoupledMPMSolverData<Real> &data, PCGCoupledMPMSolverData<Real> *dev_data, bool verbose)
    {
        if (data.fem_node_count == 0)
            return;

        CoupledFemKernel::predict_node_states<Real><<<CUDA_GRID_SIZE(data.fem_node_count), CUDA_BLOCK_SIZE>>>(dev_data);
        cudaCheck(cudaMemcpy(data.dev_fem_node_velocity_hat, data.dev_fem_node_velocity, sizeof(Real) * data.fem_node_count * 3, cudaMemcpyDeviceToDevice));

        Real prev_z_dot_r = 0;
        auto scratch_begin = thrust::device_pointer_cast(data.dev_fem_scratch);
        for (unsigned int iter = 0; iter < FEM_PCG_MAX_ITERATIONS; ++iter)
        {
            cudaCheck(cudaMemset(data.dev_fem_node_force, 0, sizeof(Real) * data.fem_node_count * 3));
            cudaCheck(cudaMemset(data.dev_fem_node_diag, 0, sizeof(Real) * data.fem_node_count * 3));
            cudaCheck(cudaMemset(data.dev_fem_node_collision_diag, 0, sizeof(Real) * data.fem_node_count * 3));

            CoupledFemKernel::assemble_tet_diagonal<Real><<<CUDA_GRID_SIZE(data.fem_tetrahedron_count), CUDA_BLOCK_SIZE>>>(dev_data);
            CoupledFemKernel::finalize_diagonal<Real><<<CUDA_GRID_SIZE(data.fem_node_count), CUDA_BLOCK_SIZE>>>(dev_data);
            CoupledFemKernel::accumulate_tet_force<Real><<<CUDA_GRID_SIZE(data.fem_tetrahedron_count), CUDA_BLOCK_SIZE>>>(dev_data);
            CoupledFemKernel::apply_ground_contact<Real><<<CUDA_GRID_SIZE(data.fem_node_count), CUDA_BLOCK_SIZE>>>(dev_data);

            CoupledFemKernel::residual_norm<Real><<<CUDA_GRID_SIZE(data.fem_node_count), CUDA_BLOCK_SIZE>>>(dev_data);
            Real residual = thrust::reduce(scratch_begin, scratch_begin + data.fem_node_count * 3);
            residual = sqrt(residual / (Real)(data.fem_node_count * 3));
            if (verbose)
                printf("FEM residual = %e\n", (double)residual);
            assert(!std::isnan(static_cast<double>(residual)));
            if (residual < (Real)FEM_RESIDUAL_TOLERANCE)
                break;

            CoupledFemKernel::preconditioned_residual<Real><<<CUDA_GRID_SIZE(data.fem_node_count), CUDA_BLOCK_SIZE>>>(dev_data);
            Real z_dot_r = thrust::reduce(scratch_begin, scratch_begin + data.fem_node_count * 3);
            Real beta = (iter > 0) ? z_dot_r / prev_z_dot_r : 0;
            prev_z_dot_r = z_dot_r;
            CoupledFemKernel::search_direction<Real><<<CUDA_GRID_SIZE(data.fem_node_count), CUDA_BLOCK_SIZE>>>(dev_data, beta);

            auto search_begin = thrust::device_pointer_cast(data.dev_fem_search_direction);
            thrust::transform(search_begin,
                              search_begin + data.fem_node_count * 3,
                              scratch_begin,
                              thrust::placeholders::_1 * thrust::placeholders::_1);
            Real p_dot_p = thrust::reduce(scratch_begin, scratch_begin + data.fem_node_count * 3);
            Real inv_norm = (p_dot_p > 0) ? (Real)(1.0 / sqrt(p_dot_p)) : 0;
            thrust::transform(search_begin,
                              search_begin + data.fem_node_count * 3,
                              scratch_begin,
                              thrust::placeholders::_1 * inv_norm);
            cudaCheck(cudaMemcpy(data.dev_fem_search_direction, data.dev_fem_scratch, sizeof(Real) * data.fem_node_count * 3, cudaMemcpyDeviceToDevice));

            cudaCheck(cudaMemset(data.dev_fem_scratch, 0, sizeof(Real) * data.fem_node_count * 3));
            CoupledFemKernel::accumulate_tet_ap<Real><<<CUDA_GRID_SIZE(data.fem_tetrahedron_count), CUDA_BLOCK_SIZE>>>(dev_data);
            CoupledFemKernel::finalize_node_pap<Real><<<CUDA_GRID_SIZE(data.fem_node_count), CUDA_BLOCK_SIZE>>>(dev_data);
            Real pAp = thrust::reduce(scratch_begin, scratch_begin + data.fem_node_count * 3);

            CoupledFemKernel::search_dot_residual<Real><<<CUDA_GRID_SIZE(data.fem_node_count), CUDA_BLOCK_SIZE>>>(dev_data);
            Real alpha = thrust::reduce(scratch_begin, scratch_begin + data.fem_node_count * 3) / pAp;

            thrust::transform(thrust::device_pointer_cast(data.dev_fem_node_velocity),
                              thrust::device_pointer_cast(data.dev_fem_node_velocity + data.fem_node_count * 3),
                              thrust::device_pointer_cast(data.dev_fem_search_direction),
                              thrust::device_pointer_cast(data.dev_fem_node_velocity),
                              thrust::placeholders::_1 + alpha * thrust::placeholders::_2);

            thrust::transform(thrust::device_pointer_cast(data.dev_fem_node_position),
                              thrust::device_pointer_cast(data.dev_fem_node_position + data.fem_node_count * 3),
                              thrust::device_pointer_cast(data.dev_fem_node_velocity),
                              thrust::device_pointer_cast(data.dev_fem_node_position_next),
                              thrust::placeholders::_1 + data.fem_time_step * thrust::placeholders::_2);
        }

        cudaCheck(cudaMemcpy(data.dev_fem_node_position, data.dev_fem_node_position_next, sizeof(Real) * data.fem_node_count * 3, cudaMemcpyDeviceToDevice));
    }

    template <typename Real>
    void RunMpmStage(PCGCoupledMPMSolverData<Real> &data, PCGCoupledMPMSolverData<Real> *dev_data, bool verbose)
    {
        if (data.mpm_particle_count == 0)
            return;

        cudaCheck(cudaMemset(data.dev_mpm_grid_momentum, 0, sizeof(Real) * data.mpm_grid_count * 3));
        cudaCheck(cudaMemset(data.dev_mpm_grid_mass, 0, sizeof(Real) * data.mpm_grid_count));
        cudaCheck(cudaMemset(data.dev_mpm_grid_diag_const, 0, sizeof(Real) * data.mpm_grid_count * 3));

        CoupledMpmKernel::particles_gravity<Real><<<CUDA_GRID_SIZE(data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(dev_data);
        CoupledMpmKernel::assign_particle_cells<Real><<<CUDA_GRID_SIZE(data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(dev_data);
        CoupledMpmKernel::p2g_momentum_mass_diag<Real><<<CUDA_GRID_SIZE(data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(dev_data);
        CoupledMpmKernel::prepare_grid_values<Real><<<CUDA_GRID_SIZE(data.mpm_grid_count), CUDA_BLOCK_SIZE>>>(dev_data);
        cudaCheck(cudaMemcpy(data.dev_mpm_grid_velocity_hat, data.dev_mpm_grid_velocity, sizeof(Real) * data.mpm_grid_count * 3, cudaMemcpyDeviceToDevice));

        Real prev_z_dot_r = 0;
        auto temp_begin = thrust::device_pointer_cast(data.dev_mpm_grid_temp);
        auto search_begin = thrust::device_pointer_cast(data.dev_mpm_grid_search);
        for (unsigned int iter = 0; iter < MPM_PCG_MAX_ITERATIONS; ++iter)
        {
            cudaCheck(cudaMemset(data.dev_mpm_grid_force, 0, sizeof(Real) * data.mpm_grid_count * 3));
            cudaCheck(cudaMemset(data.dev_mpm_grid_diag_mutable, 0, sizeof(Real) * data.mpm_grid_count * 3));

            CoupledMpmKernel::compute_particle_temp_C<Real><<<CUDA_GRID_SIZE(data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(dev_data);
            CoupledMpmKernel::p2g_grid_force<Real><<<CUDA_GRID_SIZE(data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(dev_data);
            CoupledMpmKernel::grid_boundary_force<Real><<<CUDA_GRID_SIZE(data.mpm_grid_count), CUDA_BLOCK_SIZE>>>(dev_data);

            CoupledMpmKernel::grid_residual_norm<Real><<<CUDA_GRID_SIZE(data.mpm_grid_count), CUDA_BLOCK_SIZE>>>(dev_data);
            Real residual = thrust::reduce(temp_begin, temp_begin + data.mpm_grid_count * 3);
            residual = sqrt(residual / (Real)(data.mpm_grid_count * 3));
            if (verbose)
                printf("MPM residual = %e\n", (double)residual);
            if (residual < (Real)MPM_RESIDUAL_TOLERANCE)
                break;

            CoupledMpmKernel::grid_preconditioned_residual<Real><<<CUDA_GRID_SIZE(data.mpm_grid_count), CUDA_BLOCK_SIZE>>>(dev_data);
            Real z_dot_r = thrust::reduce(temp_begin, temp_begin + data.mpm_grid_count * 3);
            Real beta = (iter > 0) ? z_dot_r / prev_z_dot_r : 0;
            prev_z_dot_r = z_dot_r;
            CoupledMpmKernel::grid_search_direction<Real><<<CUDA_GRID_SIZE(data.mpm_grid_count), CUDA_BLOCK_SIZE>>>(dev_data, beta);

            thrust::transform(search_begin,
                              search_begin + data.mpm_grid_count * 3,
                              temp_begin,
                              thrust::placeholders::_1 * thrust::placeholders::_1);
            Real p_dot_p = thrust::reduce(temp_begin, temp_begin + data.mpm_grid_count * 3);
            Real inv_norm = (p_dot_p > 0) ? (Real)(1.0 / sqrt(p_dot_p)) : 0;
            thrust::transform(search_begin,
                              search_begin + data.mpm_grid_count * 3,
                              temp_begin,
                              thrust::placeholders::_1 * inv_norm);
            cudaCheck(cudaMemcpy(data.dev_mpm_grid_search, data.dev_mpm_grid_temp, sizeof(Real) * data.mpm_grid_count * 3, cudaMemcpyDeviceToDevice));

            CoupledMpmKernel::g2p_calc_Ap_step1<Real><<<CUDA_GRID_SIZE(data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(dev_data);
            cudaCheck(cudaMemset(data.dev_mpm_grid_temp, 0, sizeof(Real) * data.mpm_grid_count * 3));
            CoupledMpmKernel::p2g_calc_Ap_step2<Real><<<CUDA_GRID_SIZE(data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(dev_data);
            CoupledMpmKernel::grid_calc_pAp_step3<Real><<<CUDA_GRID_SIZE(data.mpm_grid_count), CUDA_BLOCK_SIZE>>>(dev_data);
            Real pAp = thrust::reduce(temp_begin, temp_begin + data.mpm_grid_count * 3);

            CoupledMpmKernel::grid_search_dot_residual<Real><<<CUDA_GRID_SIZE(data.mpm_grid_count), CUDA_BLOCK_SIZE>>>(dev_data);
            Real alpha = -thrust::reduce(temp_begin, temp_begin + data.mpm_grid_count * 3) / pAp;

            thrust::transform(thrust::device_pointer_cast(data.dev_mpm_grid_velocity),
                              thrust::device_pointer_cast(data.dev_mpm_grid_velocity + data.mpm_grid_count * 3),
                              search_begin,
                              thrust::device_pointer_cast(data.dev_mpm_grid_velocity),
                              thrust::placeholders::_1 + alpha * thrust::placeholders::_2);
        }

        CoupledMpmKernel::enforce_grid_boundaries<Real><<<CUDA_GRID_SIZE(data.mpm_grid_count), CUDA_BLOCK_SIZE>>>(dev_data);

        cudaCheck(cudaMemset(data.dev_mpm_particle_velocity, 0, sizeof(Real) * data.mpm_particle_count * 3));
        cudaCheck(cudaMemset(data.dev_mpm_particle_C, 0, sizeof(Real) * data.mpm_particle_count * 9));
        CoupledMpmKernel::g2p_velocity_and_C<Real><<<CUDA_GRID_SIZE(data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(dev_data);
        CoupledMpmKernel::update_particle_F<Real><<<CUDA_GRID_SIZE(data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(dev_data);
        CoupledMpmKernel::update_particle_positions<Real><<<CUDA_GRID_SIZE(data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(dev_data);
        // CoupledMpmKernel::particles_boundary_conditions<Real><<<CUDA_GRID_SIZE(data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(dev_data);
    }
}

template <typename Real>
PCGCoupledMPMSolver<Real>::PCGCoupledMPMSolver(
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
    unsigned int boundary_thickness)
{
    assert(node_position.size() % 3 == 0);
    m_data.fem_node_count = static_cast<unsigned int>(node_position.size() / 3);
    m_data.fem_tetrahedron_count = static_cast<unsigned int>(tetrahedron.size() / 4);

    printf("PCGCoupledMPMSolver FEM: nodes=%u, tetrahedra=%u\n", m_data.fem_node_count, m_data.fem_tetrahedron_count);

    cudaCheck(cudaMalloc(&m_data.dev_fem_node_position, sizeof(Real) * node_position.size()));
    cudaCheck(cudaMemcpy(m_data.dev_fem_node_position, node_position.data(), sizeof(Real) * node_position.size(), cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc(&m_data.dev_fem_node_position_next, sizeof(Real) * node_position.size()));
    cudaCheck(cudaMalloc(&m_data.dev_fem_node_velocity, sizeof(Real) * node_position.size()));
    cudaCheck(cudaMemset(m_data.dev_fem_node_velocity, 0, sizeof(Real) * node_position.size()));
    cudaCheck(cudaMalloc(&m_data.dev_fem_node_velocity_hat, sizeof(Real) * node_position.size()));
    cudaCheck(cudaMalloc(&m_data.dev_fem_node_mass, sizeof(Real) * m_data.fem_node_count));
    cudaCheck(cudaMemset(m_data.dev_fem_node_mass, 0, sizeof(Real) * m_data.fem_node_count));
    cudaCheck(cudaMalloc(&m_data.dev_fem_node_force, sizeof(Real) * node_position.size()));
    cudaCheck(cudaMalloc(&m_data.dev_fem_node_diag, sizeof(Real) * node_position.size()));
    cudaCheck(cudaMalloc(&m_data.dev_fem_node_collision_diag, sizeof(Real) * node_position.size()));
    cudaCheck(cudaMalloc(&m_data.dev_fem_search_direction, sizeof(Real) * node_position.size()));
    cudaCheck(cudaMalloc(&m_data.dev_fem_scratch, sizeof(Real) * node_position.size()));

    if (!tetrahedron.empty())
    {
        cudaCheck(cudaMalloc(&m_data.dev_fem_tetrahedron, sizeof(unsigned int) * tetrahedron.size()));
        cudaCheck(cudaMemcpy(m_data.dev_fem_tetrahedron, tetrahedron.data(), sizeof(unsigned int) * tetrahedron.size(), cudaMemcpyHostToDevice));
        cudaCheck(cudaMalloc(&m_data.dev_fem_tet_density, sizeof(Real) * m_data.fem_tetrahedron_count));
        cudaCheck(cudaMemcpy(m_data.dev_fem_tet_density, tetrahedron_density.data(), sizeof(Real) * m_data.fem_tetrahedron_count, cudaMemcpyHostToDevice));
        cudaCheck(cudaMalloc(&m_data.dev_fem_tet_volume, sizeof(Real) * m_data.fem_tetrahedron_count));
        cudaCheck(cudaMemset(m_data.dev_fem_tet_volume, 0, sizeof(Real) * m_data.fem_tetrahedron_count));
        cudaCheck(cudaMalloc(&m_data.dev_fem_inv_dm, sizeof(Real) * m_data.fem_tetrahedron_count * 9));
    }

    m_data.surface_triangle_count = static_cast<unsigned int>(surface_triangle.size() / 3);
    if (!surface_triangle.empty())
    {
        cudaCheck(cudaMalloc(&m_data.dev_surface_triangles, sizeof(unsigned int) * surface_triangle.size()));
        cudaCheck(cudaMemcpy(m_data.dev_surface_triangles, surface_triangle.data(), sizeof(unsigned int) * surface_triangle.size(), cudaMemcpyHostToDevice));
    }

    m_data.sample_count = static_cast<unsigned int>(sample_barycentric_weights.size() / 3);
    if (m_data.sample_count > 0)
    {
        cudaCheck(cudaMalloc(&m_data.dev_sample_position, sizeof(Real) * m_data.sample_count * 3));
        cudaCheck(cudaMalloc(&m_data.dev_sample_barycentric, sizeof(Real) * sample_barycentric_weights.size()));
        cudaCheck(cudaMemcpy(m_data.dev_sample_barycentric, sample_barycentric_weights.data(), sizeof(Real) * sample_barycentric_weights.size(), cudaMemcpyHostToDevice));
        cudaCheck(cudaMalloc(&m_data.dev_sample_triangle_index, sizeof(unsigned int) * sample_triangle_idx.size()));
        cudaCheck(cudaMemcpy(m_data.dev_sample_triangle_index, sample_triangle_idx.data(), sizeof(unsigned int) * sample_triangle_idx.size(), cudaMemcpyHostToDevice));
    }

    assert(particle_position.size() % 3 == 0);
    m_data.mpm_particle_count = static_cast<unsigned int>(particle_position.size() / 3);
    printf("PCGCoupledMPMSolver MPM: particles=%u\n", m_data.mpm_particle_count);

    cudaCheck(cudaMalloc(&m_data.dev_mpm_particle_position, sizeof(Real) * particle_position.size()));
    cudaCheck(cudaMemcpy(m_data.dev_mpm_particle_position, particle_position.data(), sizeof(Real) * particle_position.size(), cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc(&m_data.dev_mpm_particle_velocity, sizeof(Real) * particle_position.size()));
    cudaCheck(cudaMemset(m_data.dev_mpm_particle_velocity, 0, sizeof(Real) * particle_position.size()));
    cudaCheck(cudaMalloc(&m_data.dev_mpm_particle_mass, sizeof(Real) * particle_mass.size()));
    cudaCheck(cudaMemcpy(m_data.dev_mpm_particle_mass, particle_mass.data(), sizeof(Real) * particle_mass.size(), cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc(&m_data.dev_mpm_particle_volume, sizeof(Real) * particle_volume.size()));
    cudaCheck(cudaMemcpy(m_data.dev_mpm_particle_volume, particle_volume.data(), sizeof(Real) * particle_volume.size(), cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc(&m_data.dev_mpm_particle_type, sizeof(unsigned int) * particle_type.size()));
    cudaCheck(cudaMemcpy(m_data.dev_mpm_particle_type, particle_type.data(), sizeof(unsigned int) * particle_type.size(), cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc(&m_data.dev_mpm_particle_C, sizeof(Real) * m_data.mpm_particle_count * 9));
    cudaCheck(cudaMemset(m_data.dev_mpm_particle_C, 0, sizeof(Real) * m_data.mpm_particle_count * 9));
    cudaCheck(cudaMalloc(&m_data.dev_mpm_particle_temp_C, sizeof(Real) * m_data.mpm_particle_count * 9));
    cudaCheck(cudaMalloc(&m_data.dev_mpm_particle_F, sizeof(Real) * m_data.mpm_particle_count * 9));
    cudaPhysics::fill_identity_matrix(m_data.dev_mpm_particle_F, m_data.mpm_particle_count, 3);
    cudaCheck(cudaMalloc(&m_data.dev_mpm_particle_cell, sizeof(unsigned int) * m_data.mpm_particle_count));
    cudaCheck(cudaMalloc(&m_data.dev_mpm_particle_scalar, sizeof(Real) * m_data.mpm_particle_count));

    std::vector<Real> inner_bbox = bbox;
    std::vector<Real> outer_bbox = bbox;
    m_data.mpm_grid_spacing = grid_spacing;
    m_data.mpm_boundary_thickness = boundary_thickness;
    for (unsigned int i = 0; i < 3; ++i)
    {
        outer_bbox[i] -= m_data.mpm_boundary_thickness * m_data.mpm_grid_spacing;
        outer_bbox[i + 3] += m_data.mpm_boundary_thickness * m_data.mpm_grid_spacing;
    }

    std::vector<unsigned int> grid_size(3);
    for (unsigned int i = 0; i < 3; ++i)
    {
        grid_size[i] = static_cast<unsigned int>(floor((outer_bbox[i + 3] - outer_bbox[i]) / m_data.mpm_grid_spacing + 0.5)) + 1;
        outer_bbox[i + 3] = outer_bbox[i] + m_data.mpm_grid_spacing * grid_size[i];
    }
    m_data.mpm_grid_count = grid_size[0] * grid_size[1] * grid_size[2];
    printf("PCGCoupledMPMSolver grid: [%u %u %u], count=%u\n", grid_size[0], grid_size[1], grid_size[2], m_data.mpm_grid_count);

    cudaCheck(cudaMalloc(&m_data.dev_mpm_inner_bbox, sizeof(Real) * 6));
    cudaCheck(cudaMemcpy(m_data.dev_mpm_inner_bbox, inner_bbox.data(), sizeof(Real) * 6, cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc(&m_data.dev_mpm_outer_bbox, sizeof(Real) * 6));
    cudaCheck(cudaMemcpy(m_data.dev_mpm_outer_bbox, outer_bbox.data(), sizeof(Real) * 6, cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc(&m_data.dev_mpm_grid_resolution, sizeof(unsigned int) * 3));
    cudaCheck(cudaMemcpy(m_data.dev_mpm_grid_resolution, grid_size.data(), sizeof(unsigned int) * 3, cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc(&m_data.dev_mpm_grid_momentum, sizeof(Real) * m_data.mpm_grid_count * 3));
    cudaCheck(cudaMemset(m_data.dev_mpm_grid_momentum, 0, sizeof(Real) * m_data.mpm_grid_count * 3));
    cudaCheck(cudaMalloc(&m_data.dev_mpm_grid_mass, sizeof(Real) * m_data.mpm_grid_count));
    cudaCheck(cudaMemset(m_data.dev_mpm_grid_mass, 0, sizeof(Real) * m_data.mpm_grid_count));
    cudaCheck(cudaMalloc(&m_data.dev_mpm_grid_force, sizeof(Real) * m_data.mpm_grid_count * 3));
    cudaCheck(cudaMemset(m_data.dev_mpm_grid_force, 0, sizeof(Real) * m_data.mpm_grid_count * 3));
    cudaCheck(cudaMalloc(&m_data.dev_mpm_grid_velocity, sizeof(Real) * m_data.mpm_grid_count * 3));
    cudaCheck(cudaMemset(m_data.dev_mpm_grid_velocity, 0, sizeof(Real) * m_data.mpm_grid_count * 3));
    cudaCheck(cudaMalloc(&m_data.dev_mpm_grid_velocity_hat, sizeof(Real) * m_data.mpm_grid_count * 3));
    cudaCheck(cudaMemset(m_data.dev_mpm_grid_velocity_hat, 0, sizeof(Real) * m_data.mpm_grid_count * 3));
    cudaCheck(cudaMalloc(&m_data.dev_mpm_grid_diag_const, sizeof(Real) * m_data.mpm_grid_count * 3));
    cudaCheck(cudaMemset(m_data.dev_mpm_grid_diag_const, 0, sizeof(Real) * m_data.mpm_grid_count * 3));
    cudaCheck(cudaMalloc(&m_data.dev_mpm_grid_diag_mutable, sizeof(Real) * m_data.mpm_grid_count * 3));
    cudaCheck(cudaMemset(m_data.dev_mpm_grid_diag_mutable, 0, sizeof(Real) * m_data.mpm_grid_count * 3));
    cudaCheck(cudaMalloc(&m_data.dev_mpm_grid_search, sizeof(Real) * m_data.mpm_grid_count * 3));
    cudaCheck(cudaMemset(m_data.dev_mpm_grid_search, 0, sizeof(Real) * m_data.mpm_grid_count * 3));
    cudaCheck(cudaMalloc(&m_data.dev_mpm_grid_temp, sizeof(Real) * m_data.mpm_grid_count * 3));
    cudaCheck(cudaMemset(m_data.dev_mpm_grid_temp, 0, sizeof(Real) * m_data.mpm_grid_count * 3));

    m_data.fem_time_step = (Real)TIME_STEP;
    m_data.fem_time_step_inv = (Real)(1.0f / TIME_STEP);
    m_data.fem_lame_mu = (Real)LAME_MU;
    m_data.fem_lame_lambda = (Real)LAME_LAMBDA;
    m_data.fem_collision_stiffness = (Real)SOLID_BOX_COLLISION_STIFFNESS;

    m_data.mpm_time_step = (Real)TIME_STEP;
    m_data.mpm_ground_stiffness = (Real)GRID_BOX_COLLISION_STIFFNESS;
    m_data.mpm_fluid_lambda = (Real)FLUID_BULK_MODULUS;
    m_data.mpm_fluid_viscosity = (Real)FLUID_VISCOSITY_COEFF;

    cudaCheck(cudaMalloc(&m_data.dev_gravity, sizeof(Real) * 3));
    std::vector<Real> gravity = {0.0f, -GRAVITY, 0.0f};
    cudaCheck(cudaMemcpy(m_data.dev_gravity, gravity.data(), sizeof(Real) * 3, cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&m_dev_data, sizeof(PCGCoupledMPMSolverData<Real>)));
    cudaCheck(cudaMemcpy(m_dev_data, &m_data, sizeof(PCGCoupledMPMSolverData<Real>), cudaMemcpyHostToDevice));

    if (m_data.fem_tetrahedron_count > 0)
    {
        CoupledFemKernel::initialize_tetrahedra<Real><<<CUDA_GRID_SIZE(m_data.fem_tetrahedron_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
        cudaCheck(cudaDeviceSynchronize());
    }
    if (m_data.sample_count > 0)
    {
        CoupledSharedKernel::update_sample_positions<Real><<<CUDA_GRID_SIZE(m_data.sample_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
        cudaCheck(cudaDeviceSynchronize());
    }
}

template <typename Real>
PCGCoupledMPMSolver<Real>::~PCGCoupledMPMSolver()
{
    cudaFree(m_data.dev_fem_node_position);
    cudaFree(m_data.dev_fem_node_position_next);
    cudaFree(m_data.dev_fem_node_velocity);
    cudaFree(m_data.dev_fem_node_velocity_hat);
    cudaFree(m_data.dev_fem_node_mass);
    cudaFree(m_data.dev_fem_node_force);
    cudaFree(m_data.dev_fem_node_diag);
    cudaFree(m_data.dev_fem_node_collision_diag);
    cudaFree(m_data.dev_fem_search_direction);
    cudaFree(m_data.dev_fem_scratch);

    cudaFree(m_data.dev_fem_tetrahedron);
    cudaFree(m_data.dev_fem_tet_density);
    cudaFree(m_data.dev_fem_tet_volume);
    cudaFree(m_data.dev_fem_inv_dm);

    cudaFree(m_data.dev_surface_triangles);
    cudaFree(m_data.dev_sample_position);
    cudaFree(m_data.dev_sample_barycentric);
    cudaFree(m_data.dev_sample_triangle_index);

    cudaFree(m_data.dev_mpm_particle_position);
    cudaFree(m_data.dev_mpm_particle_velocity);
    cudaFree(m_data.dev_mpm_particle_mass);
    cudaFree(m_data.dev_mpm_particle_volume);
    cudaFree(m_data.dev_mpm_particle_type);
    cudaFree(m_data.dev_mpm_particle_C);
    cudaFree(m_data.dev_mpm_particle_temp_C);
    cudaFree(m_data.dev_mpm_particle_F);
    cudaFree(m_data.dev_mpm_particle_cell);
    cudaFree(m_data.dev_mpm_particle_scalar);

    cudaFree(m_data.dev_mpm_inner_bbox);
    cudaFree(m_data.dev_mpm_outer_bbox);
    cudaFree(m_data.dev_mpm_grid_resolution);
    cudaFree(m_data.dev_mpm_grid_momentum);
    cudaFree(m_data.dev_mpm_grid_mass);
    cudaFree(m_data.dev_mpm_grid_force);
    cudaFree(m_data.dev_mpm_grid_velocity);
    cudaFree(m_data.dev_mpm_grid_velocity_hat);
    cudaFree(m_data.dev_mpm_grid_diag_const);
    cudaFree(m_data.dev_mpm_grid_diag_mutable);
    cudaFree(m_data.dev_mpm_grid_search);
    cudaFree(m_data.dev_mpm_grid_temp);

    cudaFree(m_data.dev_gravity);
    cudaFree(m_dev_data);
}

template <typename Real>
void PCGCoupledMPMSolver<Real>::Step()
{
    // FEM preparation
    CoupledFemKernel::predict_node_states<Real><<<CUDA_GRID_SIZE(m_data.fem_node_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaCheck(cudaMemcpy(m_data.dev_fem_node_velocity_hat, m_data.dev_fem_node_velocity, sizeof(Real) * m_data.fem_node_count * 3, cudaMemcpyDeviceToDevice));

    // MPM preparation
    cudaCheck(cudaMemset(m_data.dev_mpm_grid_momentum, 0, sizeof(Real) * m_data.mpm_grid_count * 3));
    cudaCheck(cudaMemset(m_data.dev_mpm_grid_mass, 0, sizeof(Real) * m_data.mpm_grid_count));
    cudaCheck(cudaMemset(m_data.dev_mpm_grid_diag_const, 0, sizeof(Real) * m_data.mpm_grid_count * 3));

    CoupledMpmKernel::particles_gravity<Real><<<CUDA_GRID_SIZE(m_data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
    CoupledMpmKernel::assign_particle_cells<Real><<<CUDA_GRID_SIZE(m_data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
    CoupledMpmKernel::p2g_momentum_mass_diag<Real><<<CUDA_GRID_SIZE(m_data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
    CoupledMpmKernel::prepare_grid_values<Real><<<CUDA_GRID_SIZE(m_data.mpm_grid_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaCheck(cudaMemcpy(m_data.dev_mpm_grid_velocity_hat, m_data.dev_mpm_grid_velocity, sizeof(Real) * m_data.mpm_grid_count * 3, cudaMemcpyDeviceToDevice));

    // PCG
    Real prev_z_dot_r = 0;
    auto temp_begin = thrust::device_pointer_cast(m_data.dev_mpm_grid_temp);
    auto search_begin = thrust::device_pointer_cast(m_data.dev_mpm_grid_search);
    auto scratch_begin = thrust::device_pointer_cast(m_data.dev_fem_scratch);
    for (unsigned int iter = 0; ; ++iter)
    {
        // FEM residual
        cudaCheck(cudaMemset(m_data.dev_fem_node_force, 0, sizeof(Real) * m_data.fem_node_count * 3));
        cudaCheck(cudaMemset(m_data.dev_fem_node_diag, 0, sizeof(Real) * m_data.fem_node_count * 3));
        cudaCheck(cudaMemset(m_data.dev_fem_node_collision_diag, 0, sizeof(Real) * m_data.fem_node_count * 3));

        CoupledFemKernel::assemble_tet_diagonal<Real><<<CUDA_GRID_SIZE(m_data.fem_tetrahedron_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
        CoupledFemKernel::finalize_diagonal<Real><<<CUDA_GRID_SIZE(m_data.fem_node_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
        CoupledFemKernel::accumulate_tet_force<Real><<<CUDA_GRID_SIZE(m_data.fem_tetrahedron_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
        CoupledFemKernel::apply_ground_contact<Real><<<CUDA_GRID_SIZE(m_data.fem_node_count), CUDA_BLOCK_SIZE>>>(m_dev_data);

        CoupledFemKernel::residual_norm<Real><<<CUDA_GRID_SIZE(m_data.fem_node_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
        Real fem_residual = thrust::reduce(scratch_begin, scratch_begin + m_data.fem_node_count * 3);
        fem_residual = sqrt(fem_residual / (Real)(m_data.fem_node_count * 3));
        if (m_verbose)
            printf("FEM residual = %e\n", (double)fem_residual);
        assert(!std::isnan(fem_residual));

        // MPM residual
        cudaCheck(cudaMemset(m_data.dev_mpm_grid_force, 0, sizeof(Real) * m_data.mpm_grid_count * 3));
        cudaCheck(cudaMemset(m_data.dev_mpm_grid_diag_mutable, 0, sizeof(Real) * m_data.mpm_grid_count * 3));

        CoupledMpmKernel::compute_particle_temp_C<Real><<<CUDA_GRID_SIZE(m_data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
        CoupledMpmKernel::p2g_grid_force<Real><<<CUDA_GRID_SIZE(m_data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
        CoupledMpmKernel::grid_boundary_force<Real><<<CUDA_GRID_SIZE(m_data.mpm_grid_count), CUDA_BLOCK_SIZE>>>(m_dev_data);

        CoupledMpmKernel::grid_residual_norm<Real><<<CUDA_GRID_SIZE(m_data.mpm_grid_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
        Real mpm_residual = thrust::reduce(temp_begin, temp_begin + m_data.mpm_grid_count * 3);
        mpm_residual = sqrt(mpm_residual / (Real)(m_data.mpm_grid_count * 3));
        if (m_verbose)
            printf("MPM residual = %e\n", (double)mpm_residual);
        assert(!std::isnan(mpm_residual));

        if ((mpm_residual < (Real)MPM_RESIDUAL_TOLERANCE || iter >= MPM_PCG_MAX_ITERATIONS)
            && (fem_residual < (Real)FEM_RESIDUAL_TOLERANCE || iter >= FEM_PCG_MAX_ITERATIONS))
            break;

        // PCG search direction update
        CoupledFemKernel::preconditioned_residual<Real><<<CUDA_GRID_SIZE(m_data.fem_node_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
        CoupledMpmKernel::grid_preconditioned_residual<Real><<<CUDA_GRID_SIZE(m_data.mpm_grid_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
        Real z_dot_r = thrust::reduce(scratch_begin, scratch_begin + m_data.fem_node_count * 3)
                       + thrust::reduce(temp_begin, temp_begin + m_data.mpm_grid_count * 3);
        Real beta = (iter > 0) ? z_dot_r / prev_z_dot_r : 0;
        prev_z_dot_r = z_dot_r;
        CoupledFemKernel::search_direction<Real><<<CUDA_GRID_SIZE(m_data.fem_node_count), CUDA_BLOCK_SIZE>>>(m_dev_data, beta);
        CoupledMpmKernel::grid_search_direction<Real><<<CUDA_GRID_SIZE(m_data.mpm_grid_count), CUDA_BLOCK_SIZE>>>(m_dev_data, beta);

        // Normalize search direction
        thrust::transform(thrust::device_pointer_cast(m_data.dev_fem_search_direction),
                          thrust::device_pointer_cast(m_data.dev_fem_search_direction + m_data.fem_node_count * 3),
                          scratch_begin,
                          thrust::placeholders::_1 * thrust::placeholders::_1);
        thrust::transform(search_begin,
                          search_begin + m_data.mpm_grid_count * 3,
                          temp_begin,
                          thrust::placeholders::_1 * thrust::placeholders::_1);
        Real p_dot_p = thrust::reduce(scratch_begin, scratch_begin + m_data.fem_node_count * 3)
                         + thrust::reduce(temp_begin, temp_begin + m_data.mpm_grid_count * 3);
        Real inv_norm = (p_dot_p > 0) ? (Real)(1.0 / sqrt(p_dot_p)) : 0;
        thrust::transform(thrust::device_pointer_cast(m_data.dev_fem_search_direction),
                          thrust::device_pointer_cast(m_data.dev_fem_search_direction + m_data.fem_node_count * 3),
                          scratch_begin,
                          thrust::placeholders::_1 * inv_norm);
        thrust::transform(search_begin,
                          search_begin + m_data.mpm_grid_count * 3,
                          temp_begin,
                          thrust::placeholders::_1 * inv_norm);
        cudaCheck(cudaMemcpy(m_data.dev_fem_search_direction, m_data.dev_fem_scratch, sizeof(Real) * m_data.fem_node_count * 3, cudaMemcpyDeviceToDevice));
        cudaCheck(cudaMemcpy(m_data.dev_mpm_grid_search, m_data.dev_mpm_grid_temp, sizeof(Real) * m_data.mpm_grid_count * 3, cudaMemcpyDeviceToDevice));

        // Calculate pAp
        CoupledFemKernel::accumulate_tet_ap<Real><<<CUDA_GRID_SIZE(m_data.fem_tetrahedron_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
        CoupledFemKernel::finalize_node_pap<Real><<<CUDA_GRID_SIZE(m_data.fem_node_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
        CoupledMpmKernel::g2p_calc_Ap_step1<Real><<<CUDA_GRID_SIZE(m_data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
        cudaCheck(cudaMemset(m_data.dev_mpm_grid_temp, 0, sizeof(Real) * m_data.mpm_grid_count * 3));
        CoupledMpmKernel::p2g_calc_Ap_step2<Real><<<CUDA_GRID_SIZE(m_data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
        CoupledMpmKernel::grid_calc_pAp_step3<Real><<<CUDA_GRID_SIZE(m_data.mpm_grid_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
        Real pAp = thrust::reduce(scratch_begin, scratch_begin + m_data.fem_node_count * 3)
                     + thrust::reduce(temp_begin, temp_begin + m_data.mpm_grid_count * 3);
        
        // Update solution
        CoupledFemKernel::search_dot_residual<Real><<<CUDA_GRID_SIZE(m_data.fem_node_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
        CoupledMpmKernel::grid_search_dot_residual<Real><<<CUDA_GRID_SIZE(m_data.mpm_grid_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
        Real alpha = - (thrust::reduce(scratch_begin, scratch_begin + m_data.fem_node_count * 3)
                        + thrust::reduce(temp_begin, temp_begin + m_data.mpm_grid_count * 3)) / pAp;
        thrust::transform(thrust::device_pointer_cast(m_data.dev_fem_node_velocity),
                          thrust::device_pointer_cast(m_data.dev_fem_node_velocity + m_data.fem_node_count * 3),
                          thrust::device_pointer_cast(m_data.dev_fem_search_direction),
                          thrust::device_pointer_cast(m_data.dev_fem_node_velocity),
                          thrust::placeholders::_1 + alpha * thrust::placeholders::_2);
        thrust::transform(thrust::device_pointer_cast(m_data.dev_mpm_grid_velocity),
                          thrust::device_pointer_cast(m_data.dev_mpm_grid_velocity + m_data.mpm_grid_count * 3),
                          search_begin,
                          thrust::device_pointer_cast(m_data.dev_mpm_grid_velocity),
                          thrust::placeholders::_1 + alpha * thrust::placeholders::_2);
    }

    // update FEM
    cudaCheck(cudaMemcpy(m_data.dev_fem_node_position, m_data.dev_fem_node_position_next, sizeof(Real) * m_data.fem_node_count * 3, cudaMemcpyDeviceToDevice));
    CoupledSharedKernel::update_sample_positions<Real><<<CUDA_GRID_SIZE(m_data.sample_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
    // update MPM
    CoupledMpmKernel::enforce_grid_boundaries<Real><<<CUDA_GRID_SIZE(m_data.mpm_grid_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaCheck(cudaMemset(m_data.dev_mpm_particle_velocity, 0, sizeof(Real) * m_data.mpm_particle_count * 3));
    cudaCheck(cudaMemset(m_data.dev_mpm_particle_C, 0, sizeof(Real) * m_data.mpm_particle_count * 9));
    CoupledMpmKernel::g2p_velocity_and_C<Real><<<CUDA_GRID_SIZE(m_data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
    CoupledMpmKernel::update_particle_F<Real><<<CUDA_GRID_SIZE(m_data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
    CoupledMpmKernel::update_particle_positions<Real><<<CUDA_GRID_SIZE(m_data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
    // CoupledMpmKernel::particles_boundary_conditions<Real><<<CUDA_GRID_SIZE(m_data.mpm_particle_count), CUDA_BLOCK_SIZE>>>(m_dev_data);
}

template <typename Real>
Real *PCGCoupledMPMSolver<Real>::GetDeviceNodePositions()
{
    return m_data.dev_fem_node_position;
}

template <typename Real>
Real *PCGCoupledMPMSolver<Real>::GetDeviceSamplePositions()
{
    return m_data.dev_sample_position;
}

template <typename Real>
Real *PCGCoupledMPMSolver<Real>::GetDeviceParticlePositions()
{
    return m_data.dev_mpm_particle_position;
}

template class PCGCoupledMPMSolver<float>;
template class PCGCoupledMPMSolver<double>;
template struct PCGCoupledMPMSolverData<float>;
template struct PCGCoupledMPMSolverData<double>;

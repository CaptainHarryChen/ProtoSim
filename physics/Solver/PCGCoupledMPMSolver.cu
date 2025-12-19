#include "PCGCoupledMPMSolver.cuh"
#include "CPICTools.cuh"
#include <cuda_utils/cuda_utils.cuh>
#include <Math/algebra.cuh>
#include <Math/geometry.cuh>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

namespace CoupledMPMSolverKernel
{
    template <typename Real>
    __global__ void update_sample_position(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int s = blockDim.x * blockIdx.x + threadIdx.x;
        if (s >= data.m_num_sample)
            return;
        unsigned int tri_idx = (unsigned int)data.dev_sample_tri_idx[s];
        Real pos[3] = {0, 0, 0};
        for (unsigned int v = 0; v < 3; v++)
        {
            unsigned int vert_idx = data.dev_triangle[tri_idx * 3 + v];
            Real weight = data.dev_sample_barycentric[s * 3 + v];
            cudaPhysics::axpby(pos, (Real)1, pos, weight, &data.m_fem_data.dev_vert_position[vert_idx * 3], 3);
        }
        assert(pos[0] == pos[0] && pos[1] == pos[1] && pos[2] == pos[2]); // check NaN
        cudaPhysics::vecCopy(&data.dev_sample_position[s * 3], pos, 3);
    }

    template <typename Real>
    __global__ void calc_fem_box_constraint(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data.m_fem_data.m_num_vert)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            if (data.m_fem_data.dev_vert_position[3 * v + j] < data.m_mpm_data.m_inner_bbox[j])
            {
                data.m_fem_data.dev_vert_force[3 * v + j] += 
                    data.m_fem_box_collision_stiffness
                    * (data.m_mpm_data.m_inner_bbox[j] - data.m_fem_data.dev_vert_position[3 * v + j]) 
                    * data.m_fem_data.dev_vert_mass[v];
                data.m_fem_data.dev_vert_diag_B[3 * v + j] += data.m_fem_box_collision_stiffness * data.m_time_step * data.m_fem_data.dev_vert_mass[v];
                data.m_fem_data.dev_vert_box_collision_A[3 * v + j] += data.m_fem_box_collision_stiffness * data.m_time_step * data.m_fem_data.dev_vert_mass[v];
            }
            if (data.m_fem_data.dev_vert_position[3 * v + j] > data.m_mpm_data.m_inner_bbox[j + 3])
            {
                data.m_fem_data.dev_vert_force[3 * v + j] += 
                    data.m_fem_box_collision_stiffness
                    * (data.m_mpm_data.m_inner_bbox[j + 3] - data.m_fem_data.dev_vert_position[3 * v + j]) 
                    * data.m_fem_data.dev_vert_mass[v];
                data.m_fem_data.dev_vert_diag_B[3 * v + j] += data.m_fem_box_collision_stiffness * data.m_time_step * data.m_fem_data.dev_vert_mass[v];
                data.m_fem_data.dev_vert_box_collision_A[3 * v + j] += data.m_fem_box_collision_stiffness * data.m_time_step * data.m_fem_data.dev_vert_mass[v];
            }
        }
    }

    template <typename Real>
    __global__ void calc_fem_box_constraint_pAp(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int j = blockDim.x * blockIdx.x + threadIdx.x;
        if (j >= data.m_fem_data.m_num_vert * 3)
            return;
        data.m_fem_data.dev_vert_pAp[j] += data.m_fem_data.dev_vert_box_collision_A[j] * data.m_fem_data.dev_vert_p[j] * data.m_fem_data.dev_vert_p[j];
    }

    template <typename Real>
    __global__ void calc_sample_to_leftbottom_grid_id(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int s = blockIdx.x * blockDim.x + threadIdx.x;
        if (s >= data.m_num_sample)
            return;
        auto &mpm = data.m_mpm_data;
        // find the nearest grid, so plus 0.5 and floor
        int x = floor((data.dev_sample_position[s * 3 + 0] - mpm.m_outer_bbox[0]) / mpm.m_grid_spacing + 0.5);
        int y = floor((data.dev_sample_position[s * 3 + 1] - mpm.m_outer_bbox[1]) / mpm.m_grid_spacing + 0.5);
        int z = floor((data.dev_sample_position[s * 3 + 2] - mpm.m_outer_bbox[2]) / mpm.m_grid_spacing + 0.5);
        // get the left bottom grid id
        x--;
        y--;
        z--;
        if (x < 0 || y < 0 || z < 0 || x >= mpm.m_grid_size[0] - 2 || y >= mpm.m_grid_size[1] - 2 || z >= mpm.m_grid_size[2] - 2)
        {
            x = max(0, min(x, (int)mpm.m_grid_size[0] - 3));
            y = max(0, min(y, (int)mpm.m_grid_size[1] - 3));
            z = max(0, min(z, (int)mpm.m_grid_size[2] - 3));
            printf("Warning: sample %d is out of grid, set to %d %d %d [Sample position] %.10f %.10f %.10f\n", s, x, y, z, data.dev_sample_position[s * 3 + 0], data.dev_sample_position[s * 3 + 1], data.dev_sample_position[s * 3 + 2]);
        }
        data.dev_sample_to_grid_id[s] = x * mpm.m_grid_size[1] * mpm.m_grid_size[2] + y * mpm.m_grid_size[2] + z;
    }

    template <typename Real>
    __global__ void sample_to_grid(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int s = blockIdx.x * blockDim.x + threadIdx.x;
        if (s >= data.m_num_sample)
            return;
        auto &fem = data.m_fem_data;
        auto &mpm = data.m_mpm_data;
        unsigned int tri_idx = data.dev_sample_tri_idx[s];
        const unsigned int *tri = &data.dev_triangle[tri_idx * 3];
        const Real *tri_a_pos = &fem.dev_vert_position[tri[0] * 3];
        const Real *tri_b_pos = &fem.dev_vert_position[tri[1] * 3];
        const Real *tri_c_pos = &fem.dev_vert_position[tri[2] * 3];
        unsigned int leftbottom_grid_id = data.dev_sample_to_grid_id[s];
        unsigned int x = leftbottom_grid_id / mpm.m_grid_size[1] / mpm.m_grid_size[2];
        unsigned int y = (leftbottom_grid_id / mpm.m_grid_size[2]) % mpm.m_grid_size[1];
        unsigned int z = leftbottom_grid_id % mpm.m_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * mpm.m_grid_size[1] * mpm.m_grid_size[2] + (y + dy) * mpm.m_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    CPICTools::get_grid_position(grid_position, grid_id, mpm.m_grid_size, mpm.m_outer_bbox, mpm.m_grid_spacing);
                    float distance = (float)cudaPhysics::point_to_triangle_sign_distance(grid_position, tri_a_pos, tri_b_pos, tri_c_pos);;
                    bool inside = distance < 0;
                    if (cudaPhysics::is_point_in_triangle(grid_position, tri_a_pos, tri_b_pos, tri_c_pos))
                        distance = inside ? -distance : distance;
                    else
                        distance = (float)cudaPhysics::dist3(&data.dev_sample_position[s * 3], grid_position);
                    uint64_t packed_info = CPICTools::pack_tri_info(distance, inside, tri_idx);
                    atomicMin(&data.dev_grid_tri_info[grid_id], packed_info);

                    Real weight = CPICTools::grid_particle_quadratic_weight(grid_position, &data.dev_sample_position[s * 3], mpm.m_grid_spacing);
                    atomicAdd(&data.dev_grid_sample_area[grid_id], data.dev_sample_area[s] * weight);
                }
    }

    template <typename Real>
    __global__ void G2P_particle_temp_position(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
        auto &mpm = data.m_mpm_data;
        if (p >= mpm.m_num_particle)
            return;
        const Real *particle_position = &mpm.dev_particle_position[p * 3];
        Real *particle_temp_position = &data.dev_particle_temp_position[p * 3];
        Real sum_velocity[3] = {0, 0, 0};
        unsigned int leftbottom_grid_id = mpm.dev_particle_to_grid_id[p];
        unsigned int x = leftbottom_grid_id / mpm.m_grid_size[1] / mpm.m_grid_size[2];
        unsigned int y = (leftbottom_grid_id / mpm.m_grid_size[2]) % mpm.m_grid_size[1];
        unsigned int z = leftbottom_grid_id % mpm.m_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * mpm.m_grid_size[1] * mpm.m_grid_size[2] + (y + dy) * mpm.m_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    CPICTools::get_grid_position(grid_position, grid_id, mpm.m_grid_size, mpm.m_outer_bbox, mpm.m_grid_spacing);
                    Real weight = CPICTools::grid_particle_quadratic_weight(grid_position, particle_position, mpm.m_grid_spacing);
                    cudaPhysics::axpby(sum_velocity, (Real)1, sum_velocity, weight, &mpm.dev_grid_velocity[grid_id * 3], 3);
                }
        cudaPhysics::axpby(particle_temp_position, (Real)1, particle_position, data.m_time_step, sum_velocity, 3);
    }

    template <typename Real>
    __global__ void calc_temp_particle_to_leftbottom_grid_id(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
        auto &mpm = data.m_mpm_data;
        if (p >= mpm.m_num_particle)
            return;
        // find the nearest grid, so plus 0.5 and floor
        int x = floor((data.dev_particle_temp_position[p * 3 + 0] - mpm.m_outer_bbox[0]) / mpm.m_grid_spacing + 0.5);
        int y = floor((data.dev_particle_temp_position[p * 3 + 1] - mpm.m_outer_bbox[1]) / mpm.m_grid_spacing + 0.5);
        int z = floor((data.dev_particle_temp_position[p * 3 + 2] - mpm.m_outer_bbox[2]) / mpm.m_grid_spacing + 0.5);
        // get the left bottom grid id
        x--;
        y--;
        z--;
        if (x < 0 || y < 0 || z < 0 || x >= mpm.m_grid_size[0] - 2 || y >= mpm.m_grid_size[1] - 2 || z >= mpm.m_grid_size[2] - 2)
        {
            x = max(0, min(x, (int)mpm.m_grid_size[0] - 3));
            y = max(0, min(y, (int)mpm.m_grid_size[1] - 3));
            z = max(0, min(z, (int)mpm.m_grid_size[2] - 3));
        }
        data.dev_temp_particle_to_grid_id[p] = x * mpm.m_grid_size[1] * mpm.m_grid_size[2] + y * mpm.m_grid_size[2] + z;
    }

    template <typename Real>
    __global__ void G2P_particle_sdf_and_normal(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
        auto &mpm = data.m_mpm_data;
        if (p >= mpm.m_num_particle)
            return;
        Real M[16] = {0}, Qu[4] = {0};
        M[0] = M[5] = M[10] = M[15] = mpm.m_grid_spacing / 1e6f; // to avoid singular matrix
        const Real *particle_position = &data.dev_particle_temp_position[p * 3];
        unsigned int leftbottom_grid_id = data.dev_temp_particle_to_grid_id[p]; // use temp position to calculate sdf and normal
        unsigned int x = leftbottom_grid_id / mpm.m_grid_size[1] / mpm.m_grid_size[2];
        unsigned int y = (leftbottom_grid_id / mpm.m_grid_size[2]) % mpm.m_grid_size[1];
        unsigned int z = leftbottom_grid_id % mpm.m_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * mpm.m_grid_size[1] * mpm.m_grid_size[2] + (y + dy) * mpm.m_grid_size[2] + (z + dz);
                    if (data.dev_grid_tri_info[grid_id] != 0xFFFFFFFFFFFFFFFFllu)
                    {
                        float di;
                        bool inside;
                        unsigned int tri_idx;
                        CPICTools::unpack_tri_info(data.dev_grid_tri_info[grid_id], di, inside, tri_idx);
                        Real grid_position[3], delta_position[3];
                        CPICTools::get_grid_position(grid_position, grid_id, mpm.m_grid_size, mpm.m_outer_bbox, mpm.m_grid_spacing);
                        cudaPhysics::vecSubs3(delta_position, grid_position, particle_position);
                        Real weight = CPICTools::grid_particle_quadratic_weight(grid_position, particle_position, mpm.m_grid_spacing);
                        di = inside ? -di : di;
                        M[0] += weight;
                        Qu[0] += weight * di;
                        for (int j = 0; j < 3; ++j)
                        {
                            M[j + 1] += weight * delta_position[j];
                            Qu[j + 1] += weight * di * delta_position[j];
                            for (int k = j; k < 3; ++k)
                                M[(j + 1) * 4 + (k + 1)] += weight * delta_position[j] * delta_position[k];
                        }
                    }
                }
        if (M[0] > mpm.m_grid_spacing / 1e6f)
        {
            for (int j = 0; j < 4; ++j)
                for (int k = j + 1; k < 4; ++k)
                    M[k * 4 + j] = M[j * 4 + k]; // make symmetric
            // use double precision to do matrix inverse
            double d_M[16];
            for (int ii = 0; ii < 16; ++ii)
                d_M[ii] = (double)M[ii];
            double d_M_inv[16];
            cudaPhysics::matInv4(d_M_inv, d_M);
            Real M_inv[16];
            for (int ii = 0; ii < 16; ++ii)
                M_inv[ii] = (Real)d_M_inv[ii];
            
            Real result[4];
            cudaPhysics::matVecMul(result, M_inv, Qu, 4, 4);
            data.dev_particle_dis[p] = result[0];
            cudaPhysics::norm3(&data.dev_particle_normal[p * 3], &result[1]);
        }
        else
        {
            data.dev_particle_dis[p] = 1e10f;
            data.dev_particle_normal[p * 3 + 0] = 0;
            data.dev_particle_normal[p * 3 + 1] = 0;
            data.dev_particle_normal[p * 3 + 2] = 0;
        }
    }

    template <typename Real>
    __global__ void P2G_contact_force_and_preconditioner(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
        auto &mpm = data.m_mpm_data;
        if (p >= mpm.m_num_particle || data.dev_particle_dis[p] >= 0.0f)
            return;
        Real coef = - data.m_contact_stiffness * mpm.dev_particle_volume[p] * mpm.dev_particle_F[p * 9];
        Real particle_force[3], particle_diag_B[3];
        cudaPhysics::vecMul3(particle_force, coef * data.dev_particle_dis[p], &data.dev_particle_normal[p * 3]);
        cudaPhysics::vecMul3(particle_diag_B, - coef * data.m_time_step, &data.dev_particle_normal[p * 3]);
        cudaPhysics::vecMul3(particle_diag_B, particle_diag_B, &data.dev_particle_normal[p * 3]);
        const Real *particle_position = &mpm.dev_particle_position[p * 3];
        unsigned int leftbottom_grid_id = mpm.dev_particle_to_grid_id[p];
        unsigned int x = leftbottom_grid_id / mpm.m_grid_size[1] / mpm.m_grid_size[2];
        unsigned int y = (leftbottom_grid_id / mpm.m_grid_size[2]) % mpm.m_grid_size[1];
        unsigned int z = leftbottom_grid_id % mpm.m_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * mpm.m_grid_size[1] * mpm.m_grid_size[2] + (y + dy) * mpm.m_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    CPICTools::get_grid_position(grid_position, grid_id, mpm.m_grid_size, mpm.m_outer_bbox, mpm.m_grid_spacing);
                    Real weight = CPICTools::grid_particle_quadratic_weight(grid_position, particle_position, mpm.m_grid_spacing);
                    Real force[3], diag_B[3];
                    cudaPhysics::vecMul3(force, weight, particle_force);
                    atomicAdd(&data.dev_grid_contact_force[grid_id * 3 + 0], force[0]);
                    atomicAdd(&data.dev_grid_contact_force[grid_id * 3 + 1], force[1]);
                    atomicAdd(&data.dev_grid_contact_force[grid_id * 3 + 2], force[2]);
                    cudaPhysics::vecMul3(diag_B, weight * weight, particle_diag_B);
                    atomicAdd(&mpm.dev_grid_diag_B[grid_id * 3 + 0], diag_B[0]);
                    atomicAdd(&mpm.dev_grid_diag_B[grid_id * 3 + 1], diag_B[1]);
                    atomicAdd(&mpm.dev_grid_diag_B[grid_id * 3 + 2], diag_B[2]);
                }
    }

    template <typename Real>
    __global__ void P2G_grid_nnT(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
        auto &mpm = data.m_mpm_data;
        if (p >= mpm.m_num_particle || data.dev_particle_dis[p] >= 0.0f)
            return;
        const Real *particle_position = &mpm.dev_particle_position[p * 3];
        unsigned int leftbottom_grid_id = mpm.dev_particle_to_grid_id[p];
        unsigned int x = leftbottom_grid_id / mpm.m_grid_size[1] / mpm.m_grid_size[2];
        unsigned int y = (leftbottom_grid_id / mpm.m_grid_size[2]) % mpm.m_grid_size[1];
        unsigned int z = leftbottom_grid_id % mpm.m_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * mpm.m_grid_size[1] * mpm.m_grid_size[2] + (y + dy) * mpm.m_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    CPICTools::get_grid_position(grid_position, grid_id, mpm.m_grid_size, mpm.m_outer_bbox, mpm.m_grid_spacing);
                    Real weight = CPICTools::grid_particle_quadratic_weight(grid_position, particle_position, mpm.m_grid_spacing);
                    Real nnT[9];
                    cudaPhysics::vecvecT(nnT, &data.dev_particle_normal[p * 3], &data.dev_particle_normal[p * 3], 3);
                    cudaPhysics::vecMul(nnT, weight, nnT, 9);
                    for (unsigned int j = 0; j < 9; ++j)
                        atomicAdd(&data.dev_grid_nnT[grid_id * 9 + j], nnT[j]);
                }
    }

    template <typename Real>
    __global__ void G2S2V_contact_force_and_preconditioner(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int s = blockIdx.x * blockDim.x + threadIdx.x;
        if (s >= data.m_num_sample)
            return;
        auto &mpm = data.m_mpm_data;
        unsigned int tri_idx = data.dev_sample_tri_idx[s];
        const unsigned int *tri = &data.dev_triangle[tri_idx * 3];
        unsigned int leftbottom_grid_id = data.dev_sample_to_grid_id[s];
        unsigned int x = leftbottom_grid_id / mpm.m_grid_size[1] / mpm.m_grid_size[2];
        unsigned int y = (leftbottom_grid_id / mpm.m_grid_size[2]) % mpm.m_grid_size[1];
        unsigned int z = leftbottom_grid_id % mpm.m_grid_size[2];
        Real diag_B[3] = {0}, force[3] = {0};
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * mpm.m_grid_size[1] * mpm.m_grid_size[2] + (y + dy) * mpm.m_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    CPICTools::get_grid_position(grid_position, grid_id, mpm.m_grid_size, mpm.m_outer_bbox, mpm.m_grid_spacing);
                    Real coef = CPICTools::grid_particle_quadratic_weight(grid_position, &data.dev_sample_position[s * 3], mpm.m_grid_spacing);
                    coef = data.dev_grid_sample_area[grid_id] < 1e-10 ? 0 : coef / data.dev_grid_sample_area[grid_id];
                    diag_B[0] += coef * data.dev_grid_nnT[grid_id * 9 + 0];
                    diag_B[1] += coef * data.dev_grid_nnT[grid_id * 9 + 4];
                    diag_B[2] += coef * data.dev_grid_nnT[grid_id * 9 + 8];
                    cudaPhysics::axpby(force, (Real)1, force, - coef, &data.dev_grid_contact_force[grid_id * 3], 3);
                }
        cudaPhysics::vecMul3(force, data.dev_sample_area[s]);
        cudaPhysics::vecMul3(diag_B, data.m_contact_stiffness * data.m_time_step * data.dev_sample_area[s]);
        for (unsigned int j = 0; j < 3; ++j)
        {
            unsigned int vert_idx = data.dev_triangle[tri_idx * 3 + j];
            Real bary = data.dev_sample_barycentric[s * 3 + j];
            for (unsigned int k = 0; k < 3; ++k)
            {
                atomicAdd(&data.m_fem_data.dev_vert_force[vert_idx * 3 + k], bary * force[k]);
                atomicAdd(&data.m_fem_data.dev_vert_diag_B[vert_idx * 3 + k], bary * bary * diag_B[k]);
            }
        }
    }

    template <typename Real>
    __global__ void G2P_contact_Ap_step1(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
        auto &mpm = data.m_mpm_data;
        if (p >= mpm.m_num_particle)
            return;
        Real res[3] = {0};
        const Real *particle_position = &mpm.dev_particle_position[p * 3];
        unsigned int leftbottom_grid_id = mpm.dev_particle_to_grid_id[p];
        unsigned int x = leftbottom_grid_id / mpm.m_grid_size[1] / mpm.m_grid_size[2];
        unsigned int y = (leftbottom_grid_id / mpm.m_grid_size[2]) % mpm.m_grid_size[1];
        unsigned int z = leftbottom_grid_id % mpm.m_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * mpm.m_grid_size[1] * mpm.m_grid_size[2] + (y + dy) * mpm.m_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    CPICTools::get_grid_position(grid_position, grid_id, mpm.m_grid_size, mpm.m_outer_bbox, mpm.m_grid_spacing);
                    Real weight = CPICTools::grid_particle_quadratic_weight(grid_position, particle_position, mpm.m_grid_spacing);
                    cudaPhysics::axpby(res, (Real)1, res, weight, &mpm.dev_grid_p[grid_id * 3], 3);
                }
        cudaPhysics::vecCopy(&data.dev_particle_temp3[p * 3], res, 3);
    }

    template <typename Real>
    __global__ void P2G_contact_Ap_step2(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
        auto &mpm = data.m_mpm_data;
        if (p >= mpm.m_num_particle)
            return;
        Real coef = data.m_contact_stiffness * mpm.dev_particle_volume[p] * mpm.dev_particle_F[p * 9] * data.m_time_step;
        Real dot_product = cudaPhysics::dot3(&data.dev_particle_normal[p * 3], &data.dev_particle_temp3[p * 3]);
        Real particle_Ap[3];
        cudaPhysics::vecMul3(particle_Ap, coef * dot_product, &data.dev_particle_normal[p * 3]);
        const Real *particle_position = &mpm.dev_particle_position[p * 3];
        unsigned int leftbottom_grid_id = mpm.dev_particle_to_grid_id[p];
        unsigned int x = leftbottom_grid_id / mpm.m_grid_size[1] / mpm.m_grid_size[2];
        unsigned int y = (leftbottom_grid_id / mpm.m_grid_size[2]) % mpm.m_grid_size[1];
        unsigned int z = leftbottom_grid_id % mpm.m_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * mpm.m_grid_size[1] * mpm.m_grid_size[2] + (y + dy) * mpm.m_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    CPICTools::get_grid_position(grid_position, grid_id, mpm.m_grid_size, mpm.m_outer_bbox, mpm.m_grid_spacing);
                    Real weight = CPICTools::grid_particle_quadratic_weight(grid_position, particle_position, mpm.m_grid_spacing);
                    Real grid_Ap[3];
                    cudaPhysics::vecMul3(grid_Ap, weight, particle_Ap);
                    atomicAdd(&mpm.dev_grid_temp3[grid_id * 3 + 0], grid_Ap[0]);
                    atomicAdd(&mpm.dev_grid_temp3[grid_id * 3 + 1], grid_Ap[1]);
                    atomicAdd(&mpm.dev_grid_temp3[grid_id * 3 + 2], grid_Ap[2]);
                }
    }

    template <typename Real>
    __global__ void S2G_contact_cross_Ap(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int s = blockIdx.x * blockDim.x + threadIdx.x;
        if (s >= data.m_num_sample)
            return;
        auto &mpm = data.m_mpm_data;
        auto &fem = data.m_fem_data;
        unsigned int tri_idx = data.dev_sample_tri_idx[s];
        const unsigned int *tri = &data.dev_triangle[tri_idx * 3];
        unsigned int leftbottom_grid_id = data.dev_sample_to_grid_id[s];
        unsigned int x = leftbottom_grid_id / mpm.m_grid_size[1] / mpm.m_grid_size[2];
        unsigned int y = (leftbottom_grid_id / mpm.m_grid_size[2]) % mpm.m_grid_size[1];
        unsigned int z = leftbottom_grid_id % mpm.m_grid_size[2];
        Real coef = data.m_contact_stiffness * data.m_time_step;
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * mpm.m_grid_size[1] * mpm.m_grid_size[2] + (y + dy) * mpm.m_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    CPICTools::get_grid_position(grid_position, grid_id, mpm.m_grid_size, mpm.m_outer_bbox, mpm.m_grid_spacing);
                    Real weight = CPICTools::grid_particle_quadratic_weight(grid_position, &data.dev_sample_position[s * 3], mpm.m_grid_spacing);
                    Real res[3];
                    cudaPhysics::matVec3(res, &data.dev_grid_nnT[grid_id * 9], &data.dev_sample_temp3[s * 3]);
                    cudaPhysics::vecMul3(res, coef * data.dev_sample_area[s]);
                    for (unsigned int j = 0; j < 3; ++j)
                        atomicAdd(&mpm.dev_grid_temp3[grid_id * 3 + j], res[j]);
                }
    }

    template <typename Real>
    __global__ void grid_contact_pAp_step3(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
        auto &mpm = data.m_mpm_data;
        if (j >= mpm.m_num_grid * 3)
            return;
        mpm.dev_grid_pAp[j] += mpm.dev_grid_temp3[j] * mpm.dev_grid_p[j];
    }

    template <typename Real>
    __global__ void S2V_contact_Ap_step1(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int s = blockIdx.x * blockDim.x + threadIdx.x;
        if (s >= data.m_num_sample)
            return;
        Real res[3] = {0};
        for (unsigned int j = 0; j < 3; ++j)
        {
            unsigned int vert_idx = data.dev_triangle[data.dev_sample_tri_idx[s] * 3 + j];
            Real bary = data.dev_sample_barycentric[s * 3 + j];
            cudaPhysics::axpby(res, (Real)1, res, bary, &data.m_fem_data.dev_vert_p[vert_idx * 3], 3);
        }
        cudaPhysics::vecCopy(&data.dev_sample_temp3[s * 3], res, 3);
    }

    template <typename Real>
    __global__ void G2S2V_contact_Ap_step2(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int s = blockIdx.x * blockDim.x + threadIdx.x;
        if (s >= data.m_num_sample)
            return;
        auto &mpm = data.m_mpm_data;
        auto &fem = data.m_fem_data;
        unsigned int tri_idx = data.dev_sample_tri_idx[s];
        const unsigned int *tri = &data.dev_triangle[tri_idx * 3];
        unsigned int leftbottom_grid_id = data.dev_sample_to_grid_id[s];
        unsigned int x = leftbottom_grid_id / mpm.m_grid_size[1] / mpm.m_grid_size[2];
        unsigned int y = (leftbottom_grid_id / mpm.m_grid_size[2]) % mpm.m_grid_size[1];
        unsigned int z = leftbottom_grid_id % mpm.m_grid_size[2];
        Real sample_nnT[9] = {0};
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * mpm.m_grid_size[1] * mpm.m_grid_size[2] + (y + dy) * mpm.m_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    CPICTools::get_grid_position(grid_position, grid_id, mpm.m_grid_size, mpm.m_outer_bbox, mpm.m_grid_spacing);
                    Real coef = CPICTools::grid_particle_quadratic_weight(grid_position, &data.dev_sample_position[s * 3], mpm.m_grid_spacing);
                    coef = data.dev_grid_sample_area[grid_id] < 1e-10 ? 0 : coef / data.dev_grid_sample_area[grid_id];
                    cudaPhysics::axpby(sample_nnT, (Real)1, sample_nnT, coef, &data.dev_grid_nnT[grid_id * 9], 9);
                }
        Real Ap[3];
        cudaPhysics::matVec3(Ap, sample_nnT, &data.dev_sample_temp3[s * 3]);
        cudaPhysics::vecMul3(Ap, data.m_contact_stiffness * data.m_time_step * data.dev_sample_area[s], Ap);
        for (unsigned int j = 0; j < 3; ++j)
        {
            unsigned int vert_idx = data.dev_triangle[tri_idx * 3 + j];
            Real bary = data.dev_sample_barycentric[s * 3 + j];
            for (unsigned int k = 0; k < 3; ++k)
                atomicAdd(&fem.dev_vert_temp3[vert_idx * 3 + k], bary * Ap[k]);
        }
    }

    template <typename Real>
    __global__ void G2S2V_contact_cross_Ap(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int s = blockIdx.x * blockDim.x + threadIdx.x;
        if (s >= data.m_num_sample)
            return;
        auto &mpm = data.m_mpm_data;
        auto &fem = data.m_fem_data;
        unsigned int tri_idx = data.dev_sample_tri_idx[s];
        const unsigned int *tri = &data.dev_triangle[tri_idx * 3];
        unsigned int leftbottom_grid_id = data.dev_sample_to_grid_id[s];
        unsigned int x = leftbottom_grid_id / mpm.m_grid_size[1] / mpm.m_grid_size[2];
        unsigned int y = (leftbottom_grid_id / mpm.m_grid_size[2]) % mpm.m_grid_size[1];
        unsigned int z = leftbottom_grid_id % mpm.m_grid_size[2];
        Real Ap[3] = {0};
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * mpm.m_grid_size[1] * mpm.m_grid_size[2] + (y + dy) * mpm.m_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    CPICTools::get_grid_position(grid_position, grid_id, mpm.m_grid_size, mpm.m_outer_bbox, mpm.m_grid_spacing);
                    Real coef = CPICTools::grid_particle_quadratic_weight(grid_position, &data.dev_sample_position[s * 3], mpm.m_grid_spacing);
                    coef = data.dev_grid_sample_area[grid_id] < 1e-10 ? 0 : coef / data.dev_grid_sample_area[grid_id];
                    cudaPhysics::axpby(Ap, (Real)1, Ap, coef, &mpm.dev_grid_temp3[grid_id * 3], 3);
                }
        cudaPhysics::vecMul3(Ap, data.dev_sample_area[s]);
        for (unsigned int j = 0; j < 3; ++j)
        {
            unsigned int vert_idx = data.dev_triangle[tri_idx * 3 + j];
            Real bary = data.dev_sample_barycentric[s * 3 + j];
            for (unsigned int k = 0; k < 3; ++k)
                atomicAdd(&fem.dev_vert_temp3[vert_idx * 3 + k], bary * Ap[k]);
        }
    }

    template <typename Real>
    __global__ void vert_contact_Ap_step3(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
        auto &fem = data.m_fem_data;
        if (j >= fem.m_num_vert * 3)
            return;
        fem.dev_vert_pAp[j] += fem.dev_vert_temp3[j] * fem.dev_vert_p[j];
    }

    template <typename Real>
    __global__ void update_particle_color(PCGCoupledMPMSolverData<Real> data)
    {
        unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
        auto &mpm = data.m_mpm_data;
        if (p >= mpm.m_num_particle)
            return;
        Real dis = data.dev_particle_dis[p];
        Real color[3];
        Real inside_limit = -0.1f;
        Real outside_limit = 0.5f;
        if (dis < inside_limit)
        {
            color[0] = 1.0f;
            color[1] = 0.0f;
            color[2] = 0.0f;
        }
        else if (dis < 0.0f)
        {
            color[0] = 1.0f;
            color[1] = 1.0f - dis / inside_limit;
            color[2] = 1.0f - dis / inside_limit;
        }
        else if (dis < outside_limit)
        {
            color[0] = 1.0f - dis / outside_limit;
            color[1] = 1.0f - dis / outside_limit;
            color[2] = 1.0f;
        }
        else
        {
            color[0] = 0.0f;
            color[1] = 0.0f;
            color[2] = 1.0f;
        }
        cudaPhysics::vecCopy(&data.dev_particle_color[p * 3], color, 3);
    }
}

template <typename Real>
PCGCoupledMPMSolver<Real>::PCGCoupledMPMSolver(
    const std::vector<Real> &vert_position,
    const std::vector<unsigned int> &surface_triangle,
    const std::vector<unsigned int> &tetrahedron,
    const std::vector<Real> &tetrahedron_density,

    const std::vector<Real> &sample_barycentric_weights,
    const std::vector<unsigned int> &sample_triangle_idx,
    const std::vector<Real> &sample_area,

    const std::vector<Real> &particle_position,
    const std::vector<unsigned int> &particle_type,
    const std::vector<Real> &particle_mass,
    const std::vector<Real> &particle_volume,
    std::vector<Real> bbox,
    Real grid_spacing,
    unsigned int boundary_thickness,
    const std::unordered_map<std::string, std::any> &config
) :
        m_mpm_solver(
            particle_position,
            particle_type,
            particle_mass,
            particle_volume,
            bbox,
            grid_spacing,
            boundary_thickness,
            config),
        m_fem_solver(
            vert_position,
            tetrahedron,
            tetrahedron_density,
            config)
{
    m_data.m_fem_data = m_fem_solver.m_data;
    m_data.m_mpm_data = m_mpm_solver.m_data;

    m_data.m_num_triangle = static_cast<unsigned int>(surface_triangle.size() / 3);
    m_data.m_num_sample = static_cast<unsigned int>(sample_triangle_idx.size());
    printf("PCGCoupledMPMSolver: num_triangle: %u, num_sample: %u\n", m_data.m_num_triangle, m_data.m_num_sample);

    cudaMalloc((void **)&m_data.dev_triangle, sizeof(unsigned int) * m_data.m_num_triangle * 3);
    cudaMemcpy(m_data.dev_triangle, surface_triangle.data(), sizeof(unsigned int) * m_data.m_num_triangle * 3, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&m_data.dev_sample_position, sizeof(Real) * m_data.m_num_sample * 3);
    cudaMalloc((void **)&m_data.dev_sample_barycentric, sizeof(Real) * m_data.m_num_sample * 3);
    cudaMemcpy(m_data.dev_sample_barycentric, sample_barycentric_weights.data(), sizeof(Real) * m_data.m_num_sample * 3, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_sample_area, sizeof(Real) * m_data.m_num_sample);
    cudaMemcpy(m_data.dev_sample_area, sample_area.data(), sizeof(Real) * m_data.m_num_sample, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_sample_tri_idx, sizeof(unsigned int) * m_data.m_num_sample);
    cudaMemcpy(m_data.dev_sample_tri_idx, sample_triangle_idx.data(), sizeof(unsigned int) * m_data.m_num_sample, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_sample_to_grid_id, sizeof(unsigned int) * m_data.m_num_sample);
    cudaMalloc((void **)&m_data.dev_sample_temp3, sizeof(Real) * m_data.m_num_sample * 3);

    cudaMalloc((void **)&m_data.dev_particle_color, sizeof(Real) * m_mpm_solver.m_data.m_num_particle * 3);
    cudaMalloc((void **)&m_data.dev_particle_temp_position, sizeof(Real) * m_mpm_solver.m_data.m_num_particle * 3);
    cudaMalloc((void **)&m_data.dev_temp_particle_to_grid_id, sizeof(unsigned int) * m_mpm_solver.m_data.m_num_particle);
    cudaMalloc((void **)&m_data.dev_particle_dis, sizeof(Real) * m_mpm_solver.m_data.m_num_particle);
    cudaMalloc((void **)&m_data.dev_particle_normal, sizeof(Real) * m_mpm_solver.m_data.m_num_particle * 3);
    cudaMalloc((void **)&m_data.dev_particle_temp3, sizeof(Real) * m_mpm_solver.m_data.m_num_particle * 3);

    cudaMalloc((void **)&m_data.dev_grid_tri_info, sizeof(uint64_t) * m_mpm_solver.m_data.m_num_grid);
    cudaMalloc((void **)&m_data.dev_grid_nnT, sizeof(Real) * m_mpm_solver.m_data.m_num_grid * 9);
    cudaMalloc((void **)&m_data.dev_grid_contact_force, sizeof(Real) * m_mpm_solver.m_data.m_num_grid * 3);
    cudaMalloc((void **)&m_data.dev_grid_sample_area, sizeof(Real) * m_mpm_solver.m_data.m_num_grid);

    assert(config.find("time_step") != config.end());
    m_data.m_time_step = std::any_cast<Real>(config.at("time_step"));
    m_data.m_time_step_inv = static_cast<Real>(1) / m_data.m_time_step;
    assert(config.find("line_search_max_iteration") != config.end());
    m_line_search_max_iteration = std::any_cast<unsigned int>(config.at("line_search_max_iteration"));
    assert(config.find("contact_stiffness") != config.end());
    m_data.m_contact_stiffness = std::any_cast<Real>(config.at("contact_stiffness"));
    assert(config.find("fem_box_collision_stiffness") != config.end());
    m_data.m_fem_box_collision_stiffness = std::any_cast<Real>(config.at("fem_box_collision_stiffness"));

    CoupledMPMSolverKernel::update_sample_position<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_data);
    cudaCheck(cudaDeviceSynchronize());
    printf("PCGCoupledMPMSolver initialized.\n");
}

template <typename Real>
PCGCoupledMPMSolver<Real>::~PCGCoupledMPMSolver()
{
    cudaFree(m_data.dev_triangle);

    cudaFree(m_data.dev_sample_position);
    cudaFree(m_data.dev_sample_barycentric);
    cudaFree(m_data.dev_sample_area);
    cudaFree(m_data.dev_sample_tri_idx);
    cudaFree(m_data.dev_sample_to_grid_id);
    cudaFree(m_data.dev_sample_temp3);

    cudaFree(m_data.dev_particle_temp_position);
    cudaFree(m_data.dev_temp_particle_to_grid_id);
    cudaFree(m_data.dev_particle_dis);
    cudaFree(m_data.dev_particle_normal);
    cudaFree(m_data.dev_particle_temp3);

    cudaFree(m_data.dev_grid_tri_info);
    cudaFree(m_data.dev_grid_nnT);
    cudaFree(m_data.dev_grid_contact_force);
    cudaFree(m_data.dev_grid_sample_area);
}

template <typename Real>
void PCGCoupledMPMSolver<Real>::Step()
{
    m_fem_solver.PCG_Preparation();
    m_mpm_solver.PCG_Preparation();

    Real last_residual = std::numeric_limits<Real>::max();
    Real alpha = 1.0f;
    Real prev_z_dot_r = 0.0f;
    for (unsigned int iter = 0;; ++iter)
    {
        if (m_verbose)
            printf("PCG iteration %u\n  last_residual = %e\n", iter, last_residual);
        Real residual = last_residual, fem_residual = std::numeric_limits<Real>::max(), mpm_residual = std::numeric_limits<Real>::max();
        for (unsigned int ls_iter = 0;;)
        {
            m_fem_solver.UpdateSolution(alpha);
            m_mpm_solver.UpdateSolution(alpha);
            cudaMemset(m_fem_solver.m_data.dev_vert_force, 0, sizeof(Real) * m_fem_solver.m_data.m_num_vert * 3);
            cudaMemset(m_fem_solver.m_data.dev_vert_diag_B, 0, sizeof(Real) * m_fem_solver.m_data.m_num_vert * 3);
            cudaMemset(m_mpm_solver.m_data.dev_grid_force, 0, sizeof(Real) * m_mpm_solver.m_data.m_num_grid * 3);
            cudaMemcpy(m_mpm_solver.m_data.dev_grid_diag_B, m_mpm_solver.m_data.dev_grid_diag_B_const, sizeof(Real) * m_mpm_solver.m_data.m_num_grid * 3, cudaMemcpyDeviceToDevice);
            m_fem_solver.ElasticForceAndPreconditioner();
            m_mpm_solver.MaterialForceAndPreconditioner();
            BoxConstraintForceAndPreconditioner();
            ContactConstraintForceAndPreconditioner();
            
            fem_residual = m_fem_solver.ResidualNorm();
            mpm_residual = m_mpm_solver.ResidualNorm();
            residual = fem_residual * fem_residual * m_fem_solver.m_data.m_num_vert * 3 + mpm_residual * mpm_residual * m_mpm_solver.m_data.m_num_grid * 3;
            residual = std::sqrt(residual / (m_fem_solver.m_data.m_num_vert * 3 + m_mpm_solver.m_data.m_num_grid * 3));
            if (m_verbose)
                printf("  total_residual = %e, fem_residual = %e, mpm_residual = %e\n", residual, fem_residual, mpm_residual);
            ls_iter++;
            if ((residual <= last_residual || ls_iter >= m_line_search_max_iteration) && !std::isnan(residual))
                break;
            alpha *= 0.5f;
        }
        last_residual = residual;
        cudaMemcpy(m_fem_solver.m_data.dev_vert_velocity_prev, m_fem_solver.m_data.dev_vert_velocity, sizeof(Real) * m_fem_solver.m_data.m_num_vert * 3, cudaMemcpyDeviceToDevice);
        cudaMemcpy(m_mpm_solver.m_data.dev_grid_velocity_prev, m_mpm_solver.m_data.dev_grid_velocity, sizeof(Real) * m_mpm_solver.m_data.m_num_grid * 3, cudaMemcpyDeviceToDevice);
        if ((fem_residual < m_fem_solver.m_pcg_residual_tolerance || iter >= m_fem_solver.m_pcg_max_iteration)
            && (mpm_residual < m_mpm_solver.m_pcg_residual_tolerance || iter >= m_mpm_solver.m_pcg_max_iteration))
            break;
        
        Real fem_z_dot_r = m_fem_solver.Calculate_z_dot_r();
        Real mpm_z_dot_r = m_mpm_solver.Calculate_z_dot_r();
        Real beta = iter == 0 ? 0 : (fem_z_dot_r + mpm_z_dot_r) / prev_z_dot_r;
        prev_z_dot_r = fem_z_dot_r + mpm_z_dot_r;
        m_fem_solver.SearchDirection(beta);
        m_mpm_solver.SearchDirection(beta);
        // NormalizeSearchDirection();
    
        cudaMemset(m_fem_solver.m_data.dev_vert_pAp, 0, sizeof(Real) * m_fem_solver.m_data.m_num_vert * 3);
        cudaMemset(m_mpm_solver.m_data.dev_grid_pAp, 0, sizeof(Real) * m_mpm_solver.m_data.m_num_grid * 3);
        m_fem_solver.Calc_pAp_Elastic();
        m_mpm_solver.Calc_pAp_Material();
        Calc_pAp_BoxConstraint();
        Calc_pAp_ContactConstraint();

        Real fem_pAp = m_fem_solver.Calculate_pAp();
        Real mpm_pAp = m_mpm_solver.Calculate_pAp();
        Real fem_p_dot_r = m_fem_solver.Calculate_p_dot_r();
        Real mpm_p_dot_r = m_mpm_solver.Calculate_p_dot_r();
        alpha = (fem_p_dot_r + mpm_p_dot_r) / (fem_pAp + mpm_pAp);
    }
    m_fem_solver.PCG_After();
    m_mpm_solver.PCG_After();

    CoupledMPMSolverKernel::update_sample_position<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_data);
    CoupledMPMSolverKernel::update_particle_color<Real><<<CUDA_GRID_SIZE(m_mpm_solver.m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_data);
}

template <typename Real>
Real *PCGCoupledMPMSolver<Real>::GetDeviceVertexPositions()
{
    return m_fem_solver.m_data.dev_vert_position;
}

template <typename Real>
Real *PCGCoupledMPMSolver<Real>::GetDeviceSamplePositions()
{
    return m_data.dev_sample_position;
}

template <typename Real>
Real *PCGCoupledMPMSolver<Real>::GetDeviceParticlePositions()
{
    return m_mpm_solver.m_data.dev_particle_position;
}

template <typename Real>
void PCGCoupledMPMSolver<Real>::BoxConstraintForceAndPreconditioner()
{
    cudaMemset(m_fem_solver.m_data.dev_vert_box_collision_A, 0, sizeof(Real) * m_fem_solver.m_data.m_num_vert * 3);
    CoupledMPMSolverKernel::calc_fem_box_constraint<Real><<<CUDA_GRID_SIZE(m_fem_solver.m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_data);
    m_mpm_solver.BoxConstraintForceAndPreconditioner();
}

template <typename Real>
void PCGCoupledMPMSolver<Real>::ContactConstraintForceAndPreconditioner()
{
    // MPM contact
    CoupledMPMSolverKernel::update_sample_position<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_data);
    CoupledMPMSolverKernel::calc_sample_to_leftbottom_grid_id<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_data);
    cudaMemset(m_data.dev_grid_tri_info, 0xFF, sizeof(uint64_t) * m_mpm_solver.m_data.m_num_grid);
    cudaMemset(m_data.dev_grid_sample_area, 0, sizeof(Real) * m_mpm_solver.m_data.m_num_grid);
    CoupledMPMSolverKernel::sample_to_grid<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_data);
    CoupledMPMSolverKernel::G2P_particle_temp_position<Real><<<CUDA_GRID_SIZE(m_mpm_solver.m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_data);
    CoupledMPMSolverKernel::calc_temp_particle_to_leftbottom_grid_id<Real><<<CUDA_GRID_SIZE(m_mpm_solver.m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_data);
    CoupledMPMSolverKernel::G2P_particle_sdf_and_normal<Real><<<CUDA_GRID_SIZE(m_mpm_solver.m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_data);
    cudaMemset(m_data.dev_grid_contact_force, 0, sizeof(Real) * m_mpm_solver.m_data.m_num_grid * 3);
    CoupledMPMSolverKernel::P2G_contact_force_and_preconditioner<Real><<<CUDA_GRID_SIZE(m_mpm_solver.m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_data);
    thrust::transform(thrust::device_pointer_cast(m_mpm_solver.m_data.dev_grid_force),
                      thrust::device_pointer_cast(m_mpm_solver.m_data.dev_grid_force + m_mpm_solver.m_data.m_num_grid * 3),
                      thrust::device_pointer_cast(m_data.dev_grid_contact_force),
                      thrust::device_pointer_cast(m_mpm_solver.m_data.dev_grid_force),
                      thrust::placeholders::_1 + thrust::placeholders::_2);
    // FEM contact
    cudaMemset(m_data.dev_grid_nnT, 0, sizeof(Real) * m_mpm_solver.m_data.m_num_grid * 9);
    CoupledMPMSolverKernel::P2G_grid_nnT<Real><<<CUDA_GRID_SIZE(m_mpm_solver.m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_data);
    CoupledMPMSolverKernel::G2S2V_contact_force_and_preconditioner<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_data);
}

template <typename Real>
void PCGCoupledMPMSolver<Real>::NormalizeSearchDirection()
{
    auto fem = m_fem_solver.m_data;
    auto mpm = m_mpm_solver.m_data;
    thrust::transform(thrust::device_pointer_cast(fem.dev_vert_p),
                      thrust::device_pointer_cast(fem.dev_vert_p + fem.m_num_vert * 3),
                      thrust::device_pointer_cast(fem.dev_vert_temp3),
                      thrust::placeholders::_1 * thrust::placeholders::_1);
    thrust::transform(thrust::device_pointer_cast(mpm.dev_grid_p),
                      thrust::device_pointer_cast(mpm.dev_grid_p + mpm.m_num_grid * 3),
                      thrust::device_pointer_cast(mpm.dev_grid_temp3),
                      thrust::placeholders::_1 * thrust::placeholders::_1);
    Real p_dot_p = thrust::reduce(thrust::device_pointer_cast(fem.dev_vert_temp3),
                                  thrust::device_pointer_cast(fem.dev_vert_temp3 + fem.m_num_vert * 3))
                 + thrust::reduce(thrust::device_pointer_cast(mpm.dev_grid_temp3),
                                  thrust::device_pointer_cast(mpm.dev_grid_temp3 + mpm.m_num_grid * 3));
    Real inv_sqrt_p_dot_p = 1.0f / sqrt(p_dot_p);
    thrust::transform(thrust::device_pointer_cast(fem.dev_vert_p),
                      thrust::device_pointer_cast(fem.dev_vert_p + fem.m_num_vert * 3),
                      thrust::device_pointer_cast(fem.dev_vert_temp3),
                      thrust::placeholders::_1 * inv_sqrt_p_dot_p);
    cudaMemcpy(fem.dev_vert_p, fem.dev_vert_temp3, sizeof(Real) * fem.m_num_vert * 3, cudaMemcpyDeviceToDevice);
    thrust::transform(thrust::device_pointer_cast(mpm.dev_grid_p),
                      thrust::device_pointer_cast(mpm.dev_grid_p + mpm.m_num_grid * 3),
                      thrust::device_pointer_cast(mpm.dev_grid_temp3),
                      thrust::placeholders::_1 * inv_sqrt_p_dot_p);
    cudaMemcpy(mpm.dev_grid_p, mpm.dev_grid_temp3, sizeof(Real) * mpm.m_num_grid * 3, cudaMemcpyDeviceToDevice);
}

template <typename Real>
void PCGCoupledMPMSolver<Real>::Calc_pAp_BoxConstraint()
{
    CoupledMPMSolverKernel::calc_fem_box_constraint_pAp<Real><<<CUDA_GRID_SIZE(m_fem_solver.m_data.m_num_vert * 3), CUDA_BLOCK_SIZE>>>(m_data);
    m_mpm_solver.Calc_pAp_BoxConstraint();
}

template <typename Real>
void PCGCoupledMPMSolver<Real>::Calc_pAp_ContactConstraint()
{
    // particle_temp3 = sum_i (w_ip * u_i)
    cudaMemset(m_data.dev_particle_temp3, 0, sizeof(Real) * m_mpm_solver.m_data.m_num_particle * 3);
    CoupledMPMSolverKernel::G2P_contact_Ap_step1<Real><<<CUDA_GRID_SIZE(m_mpm_solver.m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_data);
    // grid_temp3 = k * delta_t * sum_p (w_ip * V_p^n * n_p @ n_p^T @ particle_temp3)
    cudaMemset(m_mpm_solver.m_data.dev_grid_temp3, 0, sizeof(Real) * m_mpm_solver.m_data.m_num_grid * 3);
    CoupledMPMSolverKernel::P2G_contact_Ap_step2<Real><<<CUDA_GRID_SIZE(m_mpm_solver.m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_data);
    // sample_temp3 = sum_v (bary_v,s * p_v)
    CoupledMPMSolverKernel::S2V_contact_Ap_step1<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_data);
    // vert_temp3 = sum_s (bary_v,s * k * delta_t * A_s * sum_i (w_is / A_i * grid_nnT) @ sample_temp3)
    cudaMemset(m_fem_solver.m_data.dev_vert_temp3, 0, sizeof(Real) * m_fem_solver.m_data.m_num_vert * 3);
    CoupledMPMSolverKernel::G2S2V_contact_Ap_step2<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_data);
    // vert_temp3 += sum_s (bary_v,s * A_s * sum_i (w_is / A_i * grid_temp3))   [grid_temp3[i] now is (sum_j (partial f_i / partial v_j) * p_j), without cross differential]
    // CoupledMPMSolverKernel::G2S2V_contact_cross_Ap<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_data);
    // FEM contact pAp = p_v * vert_temp3
    CoupledMPMSolverKernel::vert_contact_Ap_step3<Real><<<CUDA_GRID_SIZE(m_fem_solver.m_data.m_num_vert * 3), CUDA_BLOCK_SIZE>>>(m_data);
    // grid_temp3 += k * delta_t * sum_s * (grid_nnT @ sample_temp3)  [!!!! can't swap order with above since grid_temp3 is used in cross differential]
    // CoupledMPMSolverKernel::S2G_contact_cross_Ap<Real><<<CUDA_GRID_SIZE(m_data.m_num_sample), CUDA_BLOCK_SIZE>>>(m_data);
    // MPM contact pAp = p_i * grid_temp3
    CoupledMPMSolverKernel::grid_contact_pAp_step3<Real><<<CUDA_GRID_SIZE(m_mpm_solver.m_data.m_num_grid * 3), CUDA_BLOCK_SIZE>>>(m_data);
}

template class PCGCoupledMPMSolver<float>;
template class PCGCoupledMPMSolver<double>;
template struct PCGCoupledMPMSolverData<float>;
template struct PCGCoupledMPMSolverData<double>;

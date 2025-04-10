#include "ProjectiveDynamicsSolver.cuh"
#include <cuda_utils/cuda_utils.cuh>
#include <Math/algebra.cuh>
#include <Math/elastic_model.cuh>

namespace ProjectiveDynamicsSolverKernel
{
    template <typename Real>
    __host__ __device__ __forceinline__ Real projective_dynamics_energy_cuda(Real *a, Real *b, Real *c, Real *d, Real *invDm, Real *R)
    {
        Real a0 = a[0], a1 = a[1], a2 = a[2];
        Real b0 = b[0], b1 = b[1], b2 = b[2];
        Real c0 = c[0], c1 = c[1], c2 = c[2];
        Real d0 = d[0], d1 = d[1], d2 = d[2];
        Real invDm0 = invDm[0], invDm1 = invDm[1], invDm2 = invDm[2], invDm3 = invDm[3],
             invDm4 = invDm[4], invDm5 = invDm[5], invDm6 = invDm[6], invDm7 = invDm[7], invDm8 = invDm[8];
        Real R0 = R[0], R1 = R[1], R2 = R[2], R3 = R[3], R4 = R[4], R5 = R[5], R6 = R[6], R7 = R[7], R8 = R[8];
        Real t2 = -b0;
        Real t3 = -b1;
        Real t4 = -b2;
        Real t5 = -c0;
        Real t6 = -c1;
        Real t7 = -c2;
        Real t8 = -d0;
        Real t9 = -d1;
        Real t10 = -d2;
        Real t11 = a0 + t2;
        Real t12 = a1 + t3;
        Real t13 = a2 + t4;
        Real t14 = a0 + t5;
        Real t15 = a1 + t6;
        Real t16 = a2 + t7;
        Real t17 = a0 + t8;
        Real t18 = a1 + t9;
        Real t19 = a2 + t10;
        return pow(R0 + invDm0 * t11 + invDm3 * t14 + invDm6 * t17, 2.0) + pow(R1 + invDm1 * t11 + invDm4 * t14 + invDm7 * t17, 2.0) + pow(R3 + invDm0 * t12 + invDm3 * t15 + invDm6 * t18, 2.0) + pow(R2 + invDm2 * t11 + invDm5 * t14 + invDm8 * t17, 2.0) + pow(R4 + invDm1 * t12 + invDm4 * t15 + invDm7 * t18, 2.0) + pow(R6 + invDm0 * t13 + invDm3 * t16 + invDm6 * t19, 2.0) + pow(R5 + invDm2 * t12 + invDm5 * t15 + invDm8 * t18, 2.0) + pow(R7 + invDm1 * t13 + invDm4 * t16 + invDm7 * t19, 2.0) + pow(R8 + invDm2 * t13 + invDm5 * t16 + invDm8 * t19, 2.0);
    }

    template <typename Real>
    __host__ __device__ __forceinline__ Real projective_dynamics_gradient_cuda(Real *a, Real *b, Real *c, Real *d, Real *invDm, Real *R, Real *g)
    {
        Real a0 = a[0], a1 = a[1], a2 = a[2];
        Real b0 = b[0], b1 = b[1], b2 = b[2];
        Real c0 = c[0], c1 = c[1], c2 = c[2];
        Real d0 = d[0], d1 = d[1], d2 = d[2];
        Real invDm0 = invDm[0], invDm1 = invDm[1], invDm2 = invDm[2], invDm3 = invDm[3],
             invDm4 = invDm[4], invDm5 = invDm[5], invDm6 = invDm[6], invDm7 = invDm[7], invDm8 = invDm[8];
        Real R0 = R[0], R1 = R[1], R2 = R[2], R3 = R[3], R4 = R[4], R5 = R[5], R6 = R[6], R7 = R[7], R8 = R[8];
        Real t2 = invDm0 + invDm3 + invDm6;
        Real t3 = invDm1 + invDm4 + invDm7;
        Real t4 = invDm2 + invDm5 + invDm8;
        Real t5 = -b0;
        Real t6 = -b1;
        Real t7 = -b2;
        Real t8 = -c0;
        Real t9 = -c1;
        Real t10 = -c2;
        Real t11 = -d0;
        Real t12 = -d1;
        Real t13 = -d2;
        Real t14 = a0 + t5;
        Real t15 = a1 + t6;
        Real t16 = a2 + t7;
        Real t17 = a0 + t8;
        Real t18 = a1 + t9;
        Real t19 = a2 + t10;
        Real t20 = a0 + t11;
        Real t21 = a1 + t12;
        Real t22 = a2 + t13;
        Real t23 = invDm0 * t14;
        Real t24 = invDm1 * t14;
        Real t25 = invDm0 * t15;
        Real t26 = invDm2 * t14;
        Real t27 = invDm1 * t15;
        Real t28 = invDm0 * t16;
        Real t29 = invDm2 * t15;
        Real t30 = invDm1 * t16;
        Real t31 = invDm2 * t16;
        Real t32 = invDm3 * t17;
        Real t33 = invDm4 * t17;
        Real t34 = invDm3 * t18;
        Real t35 = invDm5 * t17;
        Real t36 = invDm4 * t18;
        Real t37 = invDm3 * t19;
        Real t38 = invDm5 * t18;
        Real t39 = invDm4 * t19;
        Real t40 = invDm5 * t19;
        Real t41 = invDm6 * t20;
        Real t42 = invDm7 * t20;
        Real t43 = invDm6 * t21;
        Real t44 = invDm8 * t20;
        Real t45 = invDm7 * t21;
        Real t46 = invDm6 * t22;
        Real t47 = invDm8 * t21;
        Real t48 = invDm7 * t22;
        Real t49 = invDm8 * t22;
        Real t50 = R0 + t23 + t32 + t41;
        Real t51 = R1 + t24 + t33 + t42;
        Real t52 = R2 + t26 + t35 + t44;
        Real t53 = R3 + t25 + t34 + t43;
        Real t54 = R4 + t27 + t36 + t45;
        Real t55 = R5 + t29 + t38 + t47;
        Real t56 = R6 + t28 + t37 + t46;
        Real t57 = R7 + t30 + t39 + t48;
        Real t58 = R8 + t31 + t40 + t49;
        g[0] = t2 * t50 * 2.0 + t3 * t51 * 2.0 + t4 * t52 * 2.0;
        g[1] = t2 * t53 * 2.0 + t3 * t54 * 2.0 + t4 * t55 * 2.0;
        g[2] = t2 * t56 * 2.0 + t3 * t57 * 2.0 + t4 * t58 * 2.0;
        g[3] = invDm0 * t50 * -2.0 - invDm1 * t51 * 2.0 - invDm2 * t52 * 2.0;
        g[4] = invDm0 * t53 * -2.0 - invDm1 * t54 * 2.0 - invDm2 * t55 * 2.0;
        g[5] = invDm0 * t56 * -2.0 - invDm1 * t57 * 2.0 - invDm2 * t58 * 2.0;
        g[6] = invDm3 * t50 * -2.0 - invDm4 * t51 * 2.0 - invDm5 * t52 * 2.0;
        g[7] = invDm3 * t53 * -2.0 - invDm4 * t54 * 2.0 - invDm5 * t55 * 2.0;
        g[8] = invDm3 * t56 * -2.0 - invDm4 * t57 * 2.0 - invDm5 * t58 * 2.0;
        g[9] = invDm6 * t50 * -2.0 - invDm7 * t51 * 2.0 - invDm8 * t52 * 2.0;
        g[10] = invDm6 * t53 * -2.0 - invDm7 * t54 * 2.0 - invDm8 * t55 * 2.0;
        g[11] = invDm6 * t56 * -2.0 - invDm7 * t57 * 2.0 - invDm8 * t58 * 2.0;
    }

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
            atomicAdd(&data->dev_diag_Hessian[v[i]], rate * H[i]);
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
    __global__ void compute_ground_collision_energy(ProjectiveDynamicsSolverData<Real> *data, Real *position)
    {
        unsigned int c = blockDim.x * blockIdx.x + threadIdx.x;
        if (c >= *data->dev_ground_collision_count)
            return;
        unsigned int v = data->dev_ground_collision_ids[c];
        if (position[3 * v + 1] < 0.0f)
        {
            atomicAdd(data->dev_energy, 0.5 * data->m_stiffness * position[3 * v + 1] * position[3 * v + 1]);
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
        
        Real g[12];
        projective_dynamics_gradient_cuda<Real>(&data->dev_position[ind[0]], &data->dev_position[ind[1]], &data->dev_position[ind[2]], &data->dev_position[ind[3]], idm, R, g);
        cudaPhysics::vecMul(&data->dev_tet_force[t * 12], -data->dev_tet_volume[t] * data->m_stiffness, g, 12);
    }

    template <typename Real>
    __global__ void projective_dynamics_energy(ProjectiveDynamicsSolverData<Real> *data, Real *position)
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
        Ds[0] = position[ind[1] + 0] - position[ind[0] + 0];
        Ds[3] = position[ind[1] + 1] - position[ind[0] + 1];
        Ds[6] = position[ind[1] + 2] - position[ind[0] + 2];
        Ds[1] = position[ind[2] + 0] - position[ind[0] + 0];
        Ds[4] = position[ind[2] + 1] - position[ind[0] + 1];
        Ds[7] = position[ind[2] + 2] - position[ind[0] + 2];
        Ds[2] = position[ind[3] + 0] - position[ind[0] + 0];
        Ds[5] = position[ind[3] + 1] - position[ind[0] + 1];
        Ds[8] = position[ind[3] + 2] - position[ind[0] + 2];
        Real F[9], R[9];
        cudaPhysics::matMul3(F, Ds, idm);
        cudaPhysics::get_rotation_matrix_from_deformation_gradient(R, F);
        Real energy = projective_dynamics_energy_cuda<Real>(&position[ind[0]], &position[ind[1]], &position[ind[2]], &position[ind[3]], idm, R);
        atomicAdd(data->dev_energy, data->m_stiffness * data->dev_tet_volume[t] * energy);
    }

    template <typename Real>
    __global__ void inertia_energy(ProjectiveDynamicsSolverData<Real> *data, Real *position)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        Real diff[3];
        cudaPhysics::vecSubs3(diff, &position[3 * v], &data->dev_inertia[3 * v]);
        Real energy = 0.5 * data->dev_init_A[v] * cudaPhysics::dot3(diff, diff);
        atomicAdd(data->dev_energy, energy);
    }

    template <typename Real>
    __global__ void update_position(ProjectiveDynamicsSolverData<Real> *data, Real alpha)
    {
        unsigned int v = blockDim.x * blockIdx.x + threadIdx.x;
        if (v >= data->m_num_vert)
            return;
        cudaPhysics::axpby(&data->dev_position_next[3 * v], 3, (Real)1.0, &data->dev_position[3 * v], alpha, &data->dev_position_delta[3 * v]);
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

    cudaMalloc((void **)&m_data.dev_energy, sizeof(Real));

    cudaMalloc((void **)&m_data.dev_ground_collision_count, sizeof(unsigned int));
    cudaMalloc((void **)&m_data.dev_ground_collision_ids, sizeof(unsigned int) * m_data.m_num_vert);

    m_data.m_time_step = TIME_STEP;
    m_data.m_time_step_inv = 1.0f / m_data.m_time_step;
    m_data.m_stiffness = YOUNG_K;
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

    cudaFree(m_data.dev_energy);

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
        Real alpha = 1.;
        // if (iter % 8 == 0)
        // {
        //     alpha = line_searches();
        // }
        ProjectiveDynamicsSolverKernel::update_position<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data, alpha);
        UpdateChebysevOmega(omega, iter);
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
Real ProjectiveDynamicsSolver<Real>::line_searches()
{
    Real alpha = 1.;

    // compute energy
    Real e = compute_energy();
    unsigned int it = 0;
    while (true)
    {
        Real e_alpha = compute_energy(alpha);
        if (e > e_alpha)
        {
            break;
        }
        alpha *= BETA_LS;
        if (alpha <= ALPHA_MIN)
        {
            break;
        }
        ++it;
    }
    return alpha;
}

template <typename Real>
Real ProjectiveDynamicsSolver<Real>::compute_energy(Real alpha)
{
    Real *position_ls;
    if (alpha == 0.)
    {
        position_ls = m_data.dev_position;
    }
    else
    {
        cudaMalloc((void **)&position_ls, sizeof(Real) * m_data.m_num_vert * 3);
        cudaPhysics::axpby_kernel<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert * 3), CUDA_BLOCK_SIZE>>>(m_data.m_num_vert * 3, position_ls, 1., m_data.dev_position, alpha, m_data.dev_position_delta);
    }

    cudaMemset(m_data.dev_energy, 0, sizeof(Real));
    ProjectiveDynamicsSolverKernel::inertia_energy<Real><<<CUDA_GRID_SIZE(m_data.m_num_vert), CUDA_BLOCK_SIZE>>>(m_dev_data, position_ls);
    ProjectiveDynamicsSolverKernel::projective_dynamics_energy<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data, position_ls);
    unsigned int ground_collision_count;
    cudaMemcpy(&ground_collision_count, m_data.dev_ground_collision_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    ProjectiveDynamicsSolverKernel::compute_ground_collision_energy<Real><<<CUDA_GRID_SIZE(ground_collision_count), CUDA_BLOCK_SIZE>>>(m_dev_data, position_ls);

    if (alpha != 0.)
    {
        cudaFree(position_ls);
    }
    Real energy = 0.;
    cudaMemcpy(&energy, m_data.dev_energy, sizeof(Real), cudaMemcpyDeviceToHost);
    return energy;
}

template <typename Real>
void ProjectiveDynamicsSolver<Real>::UpdateChebysevOmega(Real &omega, unsigned int iter)
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

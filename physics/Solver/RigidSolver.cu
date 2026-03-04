#include "RigidSolver.cuh"
#include <cuda_utils/cuda_utils.cuh>
#include <Math/algebra.cuh>

namespace RigidSolverKernel
{
    template <typename Real>
    __global__ void compute_inertia_tensor(RigidSolverData<Real> data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data.num_bodies)
            return;

        Real m = data.dev_mass[i];
        data.dev_inv_mass[i] = static_cast<Real>(1.0) / m;

        int s = data.dev_shape[i];
        Real p0 = data.dev_shape_param[i * 3 + 0];
        Real p1 = data.dev_shape_param[i * 3 + 1];
        Real p2 = data.dev_shape_param[i * 3 + 2];

        Real Ixx, Iyy, Izz;

        if (s == RIGID_BODY_BOX)
        {
            Real hx = p0, hy = p1, hz = p2;
            Ixx = m * (hy * hy + hz * hz) / static_cast<Real>(3.0);
            Iyy = m * (hx * hx + hz * hz) / static_cast<Real>(3.0);
            Izz = m * (hx * hx + hy * hy) / static_cast<Real>(3.0);
        }
        else if (s == RIGID_BODY_SPHERE)
        {
            Real r = p0;
            Real I = static_cast<Real>(2.0) / static_cast<Real>(5.0) * m * r * r;
            Ixx = Iyy = Izz = I;
        }
        else
        {
            Real r = p0, hh = p1;
            Real h = static_cast<Real>(2.0) * hh;
            Real cylinder_mass = m * h / (h + static_cast<Real>(4.0) / static_cast<Real>(3.0) * r);
            Real sphere_mass = m - cylinder_mass;

            Real I_cylinder_y = cylinder_mass * r * r / static_cast<Real>(2.0);
            Real I_cylinder_xz = cylinder_mass * (static_cast<Real>(3.0) * r * r + h * h) / static_cast<Real>(12.0);

            Real I_sphere_y = static_cast<Real>(2.0) / static_cast<Real>(5.0) * sphere_mass * r * r;
            Real I_sphere_xz = I_sphere_y + sphere_mass * (hh * hh);

            Ixx = I_cylinder_xz + I_sphere_xz;
            Iyy = I_cylinder_y + I_sphere_y;
            Izz = Ixx;
        }

        Real *I = &data.dev_inertia_tensor_local[i * 9];
        I[0] = Ixx; I[1] = 0; I[2] = 0;
        I[3] = 0;   I[4] = Iyy; I[5] = 0;
        I[6] = 0;   I[7] = 0;   I[8] = Izz;

        Real *invI = &data.dev_inv_inertia_tensor_local[i * 9];
        invI[0] = static_cast<Real>(1.0) / Ixx; invI[1] = 0; invI[2] = 0;
        invI[3] = 0;   invI[4] = static_cast<Real>(1.0) / Iyy; invI[5] = 0;
        invI[6] = 0;   invI[7] = 0;   invI[8] = static_cast<Real>(1.0) / Izz;
    }

    template <typename Real>
    __global__ void compute_inv_inertia_world(RigidSolverData<Real> data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data.num_bodies)
            return;

        Real *orient = &data.dev_orientation[i * 4];
        Real *inv_I_local = &data.dev_inv_inertia_tensor_local[i * 9];
        Real *inv_I_world = &data.dev_inv_inertia_tensor_world[i * 9];

        Real R[9];
        cudaPhysics::quatToRotationMatrix(R, orient);

        Real RT[9], temp[9];
        cudaPhysics::matTrans3(RT, R);
        cudaPhysics::matMul3(temp, RT, inv_I_local);
        cudaPhysics::matMul3(inv_I_world, temp, R);
    }

    template <typename Real>
    __global__ void integrate(RigidSolverData<Real> data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data.num_bodies)
            return;

        Real *pos = &data.dev_position[i * 3];
        Real *orient = &data.dev_orientation[i * 4];
        Real *lin_vel = &data.dev_linear_velocity[i * 3];
        Real *ang_vel = &data.dev_angular_velocity[i * 3];
        Real *pos_prev = &data.dev_position_prev[i * 3];
        Real *orient_prev = &data.dev_orientation_prev[i * 4];
        Real *lin_vel_prev = &data.dev_linear_velocity_prev[i * 3];
        Real *ang_vel_prev = &data.dev_angular_velocity_prev[i * 3];

        cudaPhysics::vecCopy3(pos_prev, pos);
        cudaPhysics::vecCopy(orient_prev, orient, 4);
        cudaPhysics::vecCopy3(lin_vel_prev, lin_vel);
        cudaPhysics::vecCopy3(ang_vel_prev, ang_vel);

        cudaPhysics::axpby(lin_vel, static_cast<Real>(1.0), lin_vel, data.m_time_step, data.m_gravity, 3);
        cudaPhysics::vecMul3(lin_vel, data.m_damping, lin_vel);
        cudaPhysics::vecMul3(ang_vel, data.m_damping, ang_vel);

        cudaPhysics::axpby(pos, static_cast<Real>(1.0), pos, data.m_time_step, lin_vel, 3);
        cudaPhysics::quatAddAngularVelocity(orient, ang_vel, data.m_time_step);
    }

    template <typename Real>
    __global__ void accumulate_collision_deltas(
        RigidSolverData<Real> data,
        const CollisionInfo<Real> *collisions,
        int *dev_collision_count)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= *dev_collision_count)
            return;

        const CollisionInfo<Real> &col = collisions[i];
        if (col.body_id_a < 0)
            return;

        int body_id = col.body_id_a;

        Real *orient = &data.dev_orientation[body_id * 4];
        Real inv_m = data.dev_inv_mass[body_id];
        Real *inv_I_world = &data.dev_inv_inertia_tensor_world[body_id * 9];

        Real r_world[3];
        cudaPhysics::quatRotateVector(r_world, orient, col.local_point_a);

        Real r_cross_n[3];
        cudaPhysics::cross3(r_cross_n, r_world, col.normal);

        Real I_r_cross_n[3];
        cudaPhysics::matVec3(I_r_cross_n, inv_I_world, r_cross_n);

        Real w = inv_m + cudaPhysics::dot3(r_cross_n, I_r_cross_n);

        Real delta_lambda = col.penetration / w;

        Real delta_pos[3];
        cudaPhysics::vecMul3(delta_pos, delta_lambda * inv_m, col.normal);

        Real delta_omega[3];
        cudaPhysics::vecMul3(delta_omega, delta_lambda, I_r_cross_n);

        atomicAdd(&data.dev_delta_position[body_id * 3 + 0], delta_pos[0]);
        atomicAdd(&data.dev_delta_position[body_id * 3 + 1], delta_pos[1]);
        atomicAdd(&data.dev_delta_position[body_id * 3 + 2], delta_pos[2]);

        atomicAdd(&data.dev_delta_omega[body_id * 3 + 0], delta_omega[0]);
        atomicAdd(&data.dev_delta_omega[body_id * 3 + 1], delta_omega[1]);
        atomicAdd(&data.dev_delta_omega[body_id * 3 + 2], delta_omega[2]);

        atomicAdd(&data.dev_constraint_inv_weight[body_id], (Real)1.0);
        // printf("body: %d, penetration: %f, delta_pos: %f, %f, %f, delta_omega: %f, %f, %f\n",
        //     body_id, col.penetration, delta_pos[0], delta_pos[1], delta_pos[2], delta_omega[0], delta_omega[1], delta_omega[2]);
    }

    template <typename Real>
    __global__ void apply_deltas(RigidSolverData<Real> data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data.num_bodies)
            return;

        Real *pos = &data.dev_position[i * 3];
        Real *orient = &data.dev_orientation[i * 4];
        Real *delta_pos = &data.dev_delta_position[i * 3];
        Real weight = 1.0;
        if (data.dev_constraint_inv_weight[i] > 0)
        {
            weight = 1.0 / data.dev_constraint_inv_weight[i];
        }

        cudaPhysics::axpby(pos, (Real)1.0, pos, weight, delta_pos, 3);    

        Real delta_omega[3];
        cudaPhysics::vecMul3(delta_omega, weight, &data.dev_delta_omega[i * 3]);
        Real dq[4];
        dq[0] = -0.5 * (delta_omega[0] * orient[1] + delta_omega[1] * orient[2] + delta_omega[2] * orient[3]);
        dq[1] = 0.5 * (delta_omega[0] * orient[0] + delta_omega[1] * orient[3] - delta_omega[2] * orient[2]);
        dq[2] = 0.5 * (-delta_omega[0] * orient[3] + delta_omega[1] * orient[0] + delta_omega[2] * orient[1]);
        dq[3] = 0.5 * (delta_omega[0] * orient[2] - delta_omega[1] * orient[1] + delta_omega[2] * orient[0]);

        orient[0] += dq[0];
        orient[1] += dq[1];
        orient[2] += dq[2];
        orient[3] += dq[3];

        cudaPhysics::quatNormalize(orient);
    }

    template <typename Real>
    __global__ void update_velocity(RigidSolverData<Real> data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data.num_bodies)
            return;

        Real *pos = &data.dev_position[i * 3];
        Real *orient = &data.dev_orientation[i * 4];
        Real *lin_vel = &data.dev_linear_velocity[i * 3];
        Real *ang_vel = &data.dev_angular_velocity[i * 3];
        Real *pos_prev = &data.dev_position_prev[i * 3];
        Real *orient_prev = &data.dev_orientation_prev[i * 4];

        Real dt_inv = static_cast<Real>(1.0) / data.m_time_step;

        cudaPhysics::axpby(lin_vel, dt_inv, pos, -dt_inv, pos_prev, 3);
        // printf("body: %d, lin_vel: %f, %f, %f pos_prev: %f, %f, %f pos: %f, %f, %f\n",
        //     i, lin_vel[0], lin_vel[1], lin_vel[2], pos_prev[0], pos_prev[1], pos_prev[2], pos[0], pos[1], pos[2]);

        Real dq[4];
        cudaPhysics::vecSubs(dq, orient, orient_prev, 4);

        Real q_prev_conj[4];
        cudaPhysics::quatConjugate(q_prev_conj, orient_prev);

        Real omega_quat[4];
        cudaPhysics::quatMul(omega_quat, dq, q_prev_conj);

        ang_vel[0] = static_cast<Real>(2.0) * omega_quat[1] * dt_inv;
        ang_vel[1] = static_cast<Real>(2.0) * omega_quat[2] * dt_inv;
        ang_vel[2] = static_cast<Real>(2.0) * omega_quat[3] * dt_inv;
    }

    template <typename Real>
    __global__ void solve_contact_velocities(
        RigidSolverData<Real> data,
        const CollisionInfo<Real> *collisions,
        int *dev_collision_count)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= *dev_collision_count)
            return;

        const CollisionInfo<Real> &col = collisions[i];
        if (col.body_id_a < 0)
            return;

        int body_id = col.body_id_a;

        Real *orient = &data.dev_orientation[body_id * 4];
        Real *orient_prev = &data.dev_orientation_prev[body_id * 4];
        Real *lin_vel = &data.dev_linear_velocity[body_id * 3];
        Real *ang_vel = &data.dev_angular_velocity[body_id * 3];
        Real *lin_vel_prev = &data.dev_linear_velocity_prev[body_id * 3];
        Real *ang_vel_prev = &data.dev_angular_velocity_prev[body_id * 3];
        Real inv_m = data.dev_inv_mass[body_id];
        Real *inv_I_world = &data.dev_inv_inertia_tensor_world[body_id * 9];

        Real r_world[3];
        cudaPhysics::quatRotateVector(r_world, orient, col.local_point_a);

        Real r_cross_n[3];
        cudaPhysics::cross3(r_cross_n, r_world, col.normal);

        Real v_contact[3];
        v_contact[0] = lin_vel[0] + ang_vel[1] * r_world[2] - ang_vel[2] * r_world[1];
        v_contact[1] = lin_vel[1] + ang_vel[2] * r_world[0] - ang_vel[0] * r_world[2];
        v_contact[2] = lin_vel[2] + ang_vel[0] * r_world[1] - ang_vel[1] * r_world[0];

        Real vn = cudaPhysics::dot3(v_contact, col.normal);

        Real r_world_prev[3];
        cudaPhysics::quatRotateVector(r_world_prev, orient_prev, col.local_point_a);

        Real v_contact_prev[3];
        v_contact_prev[0] = lin_vel_prev[0] + ang_vel_prev[1] * r_world_prev[2] - ang_vel_prev[2] * r_world_prev[1];
        v_contact_prev[1] = lin_vel_prev[1] + ang_vel_prev[2] * r_world_prev[0] - ang_vel_prev[0] * r_world_prev[2];
        v_contact_prev[2] = lin_vel_prev[2] + ang_vel_prev[0] * r_world_prev[1] - ang_vel_prev[1] * r_world_prev[0];

        Real vn_prev = cudaPhysics::dot3(v_contact_prev, col.normal);

        Real restitution = data.m_restitution;
        Real gravity = -data.m_gravity[1];
        if (abs(vn) <= static_cast<Real>(2.0) * gravity * data.m_time_step)
        {
            restitution = 0;
        }

        Real vt[3];
        vt[0] = v_contact[0] - vn * col.normal[0];
        vt[1] = v_contact[1] - vn * col.normal[1];
        vt[2] = v_contact[2] - vn * col.normal[2];

        Real vt_len = cudaPhysics::len3(vt);

        Real constraint_vel_correction[3];
        constraint_vel_correction[0] = -vn * col.normal[0];
        constraint_vel_correction[1] = -vn * col.normal[1];
        constraint_vel_correction[2] = -vn * col.normal[2];

        Real bounce = -restitution * vn_prev;
        if (bounce < 0)
        {
            constraint_vel_correction[0] += bounce * col.normal[0];
            constraint_vel_correction[1] += bounce * col.normal[1];
            constraint_vel_correction[2] += bounce * col.normal[2];
        }

        if (vt_len > static_cast<Real>(1e-6))
        {
            Real vt_normalized[3];
            cudaPhysics::vecMul3(vt_normalized, static_cast<Real>(1.0) / vt_len, vt);

            Real friction_impulse = vt_len;
            Real max_friction = abs(col.penetration / data.m_time_step) * data.m_friction;
            if (friction_impulse > max_friction)
            {
                friction_impulse = max_friction;
            }

            constraint_vel_correction[0] -= vt_normalized[0] * friction_impulse;
            constraint_vel_correction[1] -= vt_normalized[1] * friction_impulse;
            constraint_vel_correction[2] -= vt_normalized[2] * friction_impulse;
        }

        Real correction_len = cudaPhysics::len3(constraint_vel_correction);
        if (correction_len < static_cast<Real>(1e-10))
            return;

        Real n[3];
        cudaPhysics::vecMul3(n, static_cast<Real>(1.0) / correction_len, constraint_vel_correction);

        Real r_cross_n_local[3];
        cudaPhysics::cross3(r_cross_n_local, r_world, n);

        Real w = inv_m + cudaPhysics::dot3(r_cross_n_local, inv_I_world) * r_cross_n_local[0] + cudaPhysics::dot3(r_cross_n_local, inv_I_world) * r_cross_n_local[1] + cudaPhysics::dot3(r_cross_n_local, inv_I_world) * r_cross_n_local[2];

        Real I_r_cross_n_local[3];
        cudaPhysics::matVec3(I_r_cross_n_local, inv_I_world, r_cross_n_local);
        w = inv_m + cudaPhysics::dot3(r_cross_n_local, I_r_cross_n_local);

        Real delta_p[3];
        cudaPhysics::vecMul3(delta_p, correction_len / w, n);

        Real delta_omega[3];
        cudaPhysics::cross3(delta_omega, r_world, delta_p);
        cudaPhysics::matVec3(delta_omega, inv_I_world, delta_omega);

        atomicAdd(&data.dev_linear_velocity[body_id * 3 + 0], delta_p[0] * inv_m);
        atomicAdd(&data.dev_linear_velocity[body_id * 3 + 1], delta_p[1] * inv_m);
        atomicAdd(&data.dev_linear_velocity[body_id * 3 + 2], delta_p[2] * inv_m);

        atomicAdd(&data.dev_angular_velocity[body_id * 3 + 0], delta_omega[0]);
        atomicAdd(&data.dev_angular_velocity[body_id * 3 + 1], delta_omega[1]);
        atomicAdd(&data.dev_angular_velocity[body_id * 3 + 2], delta_omega[2]);
    }

    template <typename Real>
    __global__ void normalize_orientation(RigidSolverData<Real> data)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data.num_bodies)
            return;

        cudaPhysics::quatNormalize(&data.dev_orientation[i * 4]);
    }
}

template <typename Real>
RigidSolver<Real>::RigidSolver(
    const std::vector<Real> &position,
    const std::vector<Real> &orientation,
    const std::vector<Real> &linear_velocity,
    const std::vector<Real> &angular_velocity,
    const std::vector<Real> &mass,
    const std::vector<int> &shape,
    const std::vector<Real> &shape_param)
    : m_ground_collision(static_cast<unsigned int>(mass.size()), static_cast<Real>(0.0))
{
    m_data.num_bodies = static_cast<unsigned int>(mass.size());

    if (m_data.num_bodies == 0)
        return;

    cudaMalloc(&m_data.dev_position, sizeof(Real) * m_data.num_bodies * 3);
    cudaMemcpy(m_data.dev_position, position.data(), sizeof(Real) * m_data.num_bodies * 3, cudaMemcpyHostToDevice);

    cudaMalloc(&m_data.dev_orientation, sizeof(Real) * m_data.num_bodies * 4);
    cudaMemcpy(m_data.dev_orientation, orientation.data(), sizeof(Real) * m_data.num_bodies * 4, cudaMemcpyHostToDevice);

    cudaMalloc(&m_data.dev_linear_velocity, sizeof(Real) * m_data.num_bodies * 3);
    cudaMemcpy(m_data.dev_linear_velocity, linear_velocity.data(), sizeof(Real) * m_data.num_bodies * 3, cudaMemcpyHostToDevice);

    cudaMalloc(&m_data.dev_angular_velocity, sizeof(Real) * m_data.num_bodies * 3);
    cudaMemcpy(m_data.dev_angular_velocity, angular_velocity.data(), sizeof(Real) * m_data.num_bodies * 3, cudaMemcpyHostToDevice);

    cudaMalloc(&m_data.dev_mass, sizeof(Real) * m_data.num_bodies);
    cudaMemcpy(m_data.dev_mass, mass.data(), sizeof(Real) * m_data.num_bodies, cudaMemcpyHostToDevice);

    cudaMalloc(&m_data.dev_inv_mass, sizeof(Real) * m_data.num_bodies);

    cudaMalloc(&m_data.dev_inertia_tensor_local, sizeof(Real) * m_data.num_bodies * 9);
    cudaMalloc(&m_data.dev_inv_inertia_tensor_local, sizeof(Real) * m_data.num_bodies * 9);
    cudaMalloc(&m_data.dev_inv_inertia_tensor_world, sizeof(Real) * m_data.num_bodies * 9);

    cudaMalloc(&m_data.dev_shape, sizeof(int) * m_data.num_bodies);
    cudaMemcpy(m_data.dev_shape, shape.data(), sizeof(int) * m_data.num_bodies, cudaMemcpyHostToDevice);

    cudaMalloc(&m_data.dev_shape_param, sizeof(Real) * m_data.num_bodies * 3);
    cudaMemcpy(m_data.dev_shape_param, shape_param.data(), sizeof(Real) * m_data.num_bodies * 3, cudaMemcpyHostToDevice);

    cudaMalloc(&m_data.dev_position_prev, sizeof(Real) * m_data.num_bodies * 3);
    cudaMalloc(&m_data.dev_orientation_prev, sizeof(Real) * m_data.num_bodies * 4);
    cudaMalloc(&m_data.dev_linear_velocity_prev, sizeof(Real) * m_data.num_bodies * 3);
    cudaMalloc(&m_data.dev_angular_velocity_prev, sizeof(Real) * m_data.num_bodies * 3);

    cudaMalloc(&m_data.dev_delta_position, sizeof(Real) * m_data.num_bodies * 3);
    cudaMalloc(&m_data.dev_delta_omega, sizeof(Real) * m_data.num_bodies * 3);
    cudaMalloc(&m_data.dev_constraint_inv_weight, sizeof(Real) * m_data.num_bodies);

    RigidSolverKernel::compute_inertia_tensor<Real><<<CUDA_GRID_SIZE(m_data.num_bodies), CUDA_BLOCK_SIZE>>>(m_data);

    m_data.m_time_step = static_cast<Real>(0.01);
    m_data.m_gravity[0] = static_cast<Real>(0.0);
    m_data.m_gravity[1] = static_cast<Real>(-9.8);
    m_data.m_gravity[2] = static_cast<Real>(0.0);
    m_data.m_damping = static_cast<Real>(1.0);
    m_data.m_restitution = static_cast<Real>(0.3);
    m_data.m_friction = static_cast<Real>(0.5);
    m_data.m_num_substeps = 4;
    m_data.m_num_solver_iterations = 100;
}

template <typename Real>
RigidSolver<Real>::~RigidSolver()
{
    if (m_data.dev_position)
        cudaFree(m_data.dev_position);
    if (m_data.dev_orientation)
        cudaFree(m_data.dev_orientation);
    if (m_data.dev_linear_velocity)
        cudaFree(m_data.dev_linear_velocity);
    if (m_data.dev_angular_velocity)
        cudaFree(m_data.dev_angular_velocity);
    if (m_data.dev_mass)
        cudaFree(m_data.dev_mass);
    if (m_data.dev_inv_mass)
        cudaFree(m_data.dev_inv_mass);
    if (m_data.dev_inertia_tensor_local)
        cudaFree(m_data.dev_inertia_tensor_local);
    if (m_data.dev_inv_inertia_tensor_local)
        cudaFree(m_data.dev_inv_inertia_tensor_local);
    if (m_data.dev_inv_inertia_tensor_world)
        cudaFree(m_data.dev_inv_inertia_tensor_world);
    if (m_data.dev_shape)
        cudaFree(m_data.dev_shape);
    if (m_data.dev_shape_param)
        cudaFree(m_data.dev_shape_param);
    if (m_data.dev_position_prev)
        cudaFree(m_data.dev_position_prev);
    if (m_data.dev_orientation_prev)
        cudaFree(m_data.dev_orientation_prev);
    if (m_data.dev_linear_velocity_prev)
        cudaFree(m_data.dev_linear_velocity_prev);
    if (m_data.dev_angular_velocity_prev)
        cudaFree(m_data.dev_angular_velocity_prev);
    if (m_data.dev_delta_position)
        cudaFree(m_data.dev_delta_position);
    if (m_data.dev_delta_omega)
        cudaFree(m_data.dev_delta_omega);
    if (m_data.dev_constraint_inv_weight)
        cudaFree(m_data.dev_constraint_inv_weight);
}

template <typename Real>
void RigidSolver<Real>::Step()
{
    Real substep_dt = m_data.m_time_step / static_cast<Real>(m_data.m_num_substeps);
    Real original_dt = m_data.m_time_step;
    m_data.m_time_step = substep_dt;

    for (unsigned int substep = 0; substep < m_data.m_num_substeps; ++substep)
    {
        RigidSolverKernel::integrate<Real><<<CUDA_GRID_SIZE(m_data.num_bodies), CUDA_BLOCK_SIZE>>>(m_data);
        RigidSolverKernel::compute_inv_inertia_world<Real><<<CUDA_GRID_SIZE(m_data.num_bodies), CUDA_BLOCK_SIZE>>>(m_data);

        m_ground_collision.Detect(
            m_data.dev_position,
            m_data.dev_orientation,
            m_data.dev_shape,
            m_data.dev_shape_param);

        for (unsigned int iter = 0; iter < m_data.m_num_solver_iterations; ++iter)
        {
            cudaMemset(m_data.dev_delta_position, 0, sizeof(Real) * m_data.num_bodies * 3);
            cudaMemset(m_data.dev_delta_omega, 0, sizeof(Real) * m_data.num_bodies * 3);
            cudaMemset(m_data.dev_constraint_inv_weight, 0, sizeof(Real) * m_data.num_bodies);
            RigidSolverKernel::accumulate_collision_deltas<Real><<<CUDA_GRID_SIZE(m_ground_collision.GetData().max_collisions), CUDA_BLOCK_SIZE>>>(
                m_data, m_ground_collision.GetData().dev_collisions, m_ground_collision.GetData().dev_collision_count);
            RigidSolverKernel::apply_deltas<Real><<<CUDA_GRID_SIZE(m_data.num_bodies), CUDA_BLOCK_SIZE>>>(m_data);
            RigidSolverKernel::compute_inv_inertia_world<Real><<<CUDA_GRID_SIZE(m_data.num_bodies), CUDA_BLOCK_SIZE>>>(m_data);
        }

        RigidSolverKernel::update_velocity<Real><<<CUDA_GRID_SIZE(m_data.num_bodies), CUDA_BLOCK_SIZE>>>(m_data);

        RigidSolverKernel::solve_contact_velocities<Real><<<CUDA_GRID_SIZE(m_ground_collision.GetData().max_collisions), CUDA_BLOCK_SIZE>>>(
            m_data, m_ground_collision.GetData().dev_collisions, m_ground_collision.GetData().dev_collision_count);

        RigidSolverKernel::normalize_orientation<Real><<<CUDA_GRID_SIZE(m_data.num_bodies), CUDA_BLOCK_SIZE>>>(m_data);
    }

    m_data.m_time_step = original_dt;
}

template <typename Real>
Real *RigidSolver<Real>::GetDevicePositions()
{
    return m_data.dev_position;
}

template <typename Real>
Real *RigidSolver<Real>::GetDeviceOrientations()
{
    return m_data.dev_orientation;
}

template struct RigidSolverData<float>;
template struct RigidSolverData<double>;
template class RigidSolver<float>;
template class RigidSolver<double>;

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
        Real *lin_vel_prev = &data.dev_linear_velocity_before_constraint[i * 3];
        Real *ang_vel_prev = &data.dev_angular_velocity_before_constraint[i * 3];

        cudaPhysics::vecCopy3(pos_prev, pos);
        cudaPhysics::vecCopy(orient_prev, orient, 4);

        cudaPhysics::axpby(lin_vel, static_cast<Real>(1.0), lin_vel, data.m_time_step, data.m_gravity, 3);
        cudaPhysics::vecMul3(lin_vel, data.m_damping, lin_vel);
        cudaPhysics::vecMul3(ang_vel, data.m_damping, ang_vel);

        cudaPhysics::vecCopy3(lin_vel_prev, lin_vel);
        cudaPhysics::vecCopy3(ang_vel_prev, ang_vel);

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

        int body_id_a = col.body_id_a;
        int body_id_b = col.body_id_b;

        Real *pos_a = &data.dev_position[body_id_a * 3];
        Real *orient_a = &data.dev_orientation[body_id_a * 4];
        Real inv_m_a = data.dev_inv_mass[body_id_a];
        Real *inv_I_world_a = &data.dev_inv_inertia_tensor_world[body_id_a * 9];

        Real r_world_a[3];
        cudaPhysics::quatRotateVector(r_world_a, orient_a, col.local_point_a);
        Real world_point_a[3];
        cudaPhysics::vecAdd3(world_point_a, pos_a, r_world_a);

        Real world_point_b[3];
        Real r_world_b[3];
        Real inv_m_b = static_cast<Real>(0.0);
        Real *inv_I_world_b = nullptr;

        if (body_id_b < 0)
        {
            cudaPhysics::vecCopy3(world_point_b, col.local_point_b);
        }
        else
        {
            Real *pos_b = &data.dev_position[body_id_b * 3];
            Real *orient_b = &data.dev_orientation[body_id_b * 4];
            inv_m_b = data.dev_inv_mass[body_id_b];
            inv_I_world_b = &data.dev_inv_inertia_tensor_world[body_id_b * 9];

            cudaPhysics::quatRotateVector(r_world_b, orient_b, col.local_point_b);
            cudaPhysics::vecAdd3(world_point_b, pos_b, r_world_b);
        }

        Real a_to_b[3];
        cudaPhysics::vecSubs3(a_to_b, world_point_a, world_point_b);
        Real penetration = cudaPhysics::dot3(col.normal, a_to_b);

        printf("i:%d, body_id_a:%d, body_id_b:%d, world_point_a:(%f, %f, %f), world_point_b:(%f, %f, %f), normal:(%f, %f, %f), penetration:%f\n",
            i, body_id_a, body_id_b, world_point_a[0], world_point_a[1], world_point_a[2],
            world_point_b[0], world_point_b[1], world_point_b[2], 
             col.normal[0], col.normal[1], col.normal[2], penetration);

        if (penetration <= static_cast<Real>(0.0))
            return;

        Real r_cross_n_a[3];
        cudaPhysics::cross3(r_cross_n_a, col.normal,  r_world_a);

        Real I_r_cross_n_a[3];
        cudaPhysics::matVec3(I_r_cross_n_a, inv_I_world_a, r_cross_n_a);

        Real w_a = inv_m_a + cudaPhysics::dot3(r_cross_n_a, I_r_cross_n_a);

        Real w_b = static_cast<Real>(0.0);
        Real I_r_cross_n_b[3] = {static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)};
        if (body_id_b >= 0)
        {
            Real r_cross_n_b[3];
            Real neg_normal[3];
            cudaPhysics::vecMul3(neg_normal, static_cast<Real>(-1.0), col.normal);
            cudaPhysics::cross3(r_cross_n_b, r_world_b, neg_normal);

            cudaPhysics::matVec3(I_r_cross_n_b, inv_I_world_b, r_cross_n_b);
            w_b = inv_m_b + cudaPhysics::dot3(r_cross_n_b, I_r_cross_n_b);
        }

        Real w_total = w_a + w_b;
        if (w_total < static_cast<Real>(1e-10))
            return;

        Real delta_lambda = penetration / w_total;

        Real delta_pos_a[3];
        cudaPhysics::vecMul3(delta_pos_a, -delta_lambda * inv_m_a, col.normal);

        Real delta_omega_a[3];
        cudaPhysics::vecMul3(delta_omega_a, delta_lambda, I_r_cross_n_a);

        atomicAdd(&data.dev_delta_position[body_id_a * 3 + 0], delta_pos_a[0]);
        atomicAdd(&data.dev_delta_position[body_id_a * 3 + 1], delta_pos_a[1]);
        atomicAdd(&data.dev_delta_position[body_id_a * 3 + 2], delta_pos_a[2]);

        atomicAdd(&data.dev_delta_omega[body_id_a * 3 + 0], delta_omega_a[0]);
        atomicAdd(&data.dev_delta_omega[body_id_a * 3 + 1], delta_omega_a[1]);
        atomicAdd(&data.dev_delta_omega[body_id_a * 3 + 2], delta_omega_a[2]);

        atomicAdd(&data.dev_constraint_inv_weight[body_id_a], (Real)1.0);

        if (body_id_b >= 0)
        {
            Real delta_pos_b[3];
            cudaPhysics::vecMul3(delta_pos_b, delta_lambda * inv_m_b, col.normal);

            Real delta_omega_b[3];
            cudaPhysics::vecMul3(delta_omega_b, -delta_lambda, I_r_cross_n_b);

            atomicAdd(&data.dev_delta_position[body_id_b * 3 + 0], delta_pos_b[0]);
            atomicAdd(&data.dev_delta_position[body_id_b * 3 + 1], delta_pos_b[1]);
            atomicAdd(&data.dev_delta_position[body_id_b * 3 + 2], delta_pos_b[2]);

            atomicAdd(&data.dev_delta_omega[body_id_b * 3 + 0], delta_omega_b[0]);
            atomicAdd(&data.dev_delta_omega[body_id_b * 3 + 1], delta_omega_b[1]);
            atomicAdd(&data.dev_delta_omega[body_id_b * 3 + 2], delta_omega_b[2]);

            atomicAdd(&data.dev_constraint_inv_weight[body_id_b], (Real)1.0);
        }
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
            weight = 1.0 / data.dev_constraint_inv_weight[i];

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

        int body_id_a = col.body_id_a;
        int body_id_b = col.body_id_b;

        Real *pos_a = &data.dev_position[body_id_a * 3];
        Real *orient_a = &data.dev_orientation[body_id_a * 4];
        Real *orient_prev_a = &data.dev_orientation_prev[body_id_a * 4];
        Real *lin_vel_a = &data.dev_linear_velocity[body_id_a * 3];
        Real *ang_vel_a = &data.dev_angular_velocity[body_id_a * 3];
        Real *lin_vel_prev_a = &data.dev_linear_velocity_before_constraint[body_id_a * 3];
        Real *ang_vel_prev_a = &data.dev_angular_velocity_before_constraint[body_id_a * 3];
        Real inv_m_a = data.dev_inv_mass[body_id_a];
        Real *inv_I_world_a = &data.dev_inv_inertia_tensor_world[body_id_a * 9];

        Real r_world_a[3];
        cudaPhysics::quatRotateVector(r_world_a, orient_a, col.local_point_a);

        Real v_contact_a[3];
        v_contact_a[0] = lin_vel_a[0] + ang_vel_a[1] * r_world_a[2] - ang_vel_a[2] * r_world_a[1];
        v_contact_a[1] = lin_vel_a[1] + ang_vel_a[2] * r_world_a[0] - ang_vel_a[0] * r_world_a[2];
        v_contact_a[2] = lin_vel_a[2] + ang_vel_a[0] * r_world_a[1] - ang_vel_a[1] * r_world_a[0];

        Real r_world_prev_a[3];
        cudaPhysics::quatRotateVector(r_world_prev_a, orient_prev_a, col.local_point_a);

        Real v_contact_prev_a[3];
        v_contact_prev_a[0] = lin_vel_prev_a[0] + ang_vel_prev_a[1] * r_world_prev_a[2] - ang_vel_prev_a[2] * r_world_prev_a[1];
        v_contact_prev_a[1] = lin_vel_prev_a[1] + ang_vel_prev_a[2] * r_world_prev_a[0] - ang_vel_prev_a[0] * r_world_prev_a[2];
        v_contact_prev_a[2] = lin_vel_prev_a[2] + ang_vel_prev_a[0] * r_world_prev_a[1] - ang_vel_prev_a[1] * r_world_prev_a[0];

        Real v_contact_b[3] = {static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)};
        Real v_contact_prev_b[3] = {static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)};
        Real r_world_b[3] = {static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)};
        Real inv_m_b = static_cast<Real>(0.0);
        Real *inv_I_world_b = nullptr;

        if (body_id_b >= 0)
        {
            Real *pos_b = &data.dev_position[body_id_b * 3];
            Real *orient_b = &data.dev_orientation[body_id_b * 4];
            Real *orient_prev_b = &data.dev_orientation_prev[body_id_b * 4];
            Real *lin_vel_b = &data.dev_linear_velocity[body_id_b * 3];
            Real *ang_vel_b = &data.dev_angular_velocity[body_id_b * 3];
            Real *lin_vel_prev_b = &data.dev_linear_velocity_before_constraint[body_id_b * 3];
            Real *ang_vel_prev_b = &data.dev_angular_velocity_before_constraint[body_id_b * 3];
            inv_m_b = data.dev_inv_mass[body_id_b];
            inv_I_world_b = &data.dev_inv_inertia_tensor_world[body_id_b * 9];

            cudaPhysics::quatRotateVector(r_world_b, orient_b, col.local_point_b);

            v_contact_b[0] = lin_vel_b[0] + ang_vel_b[1] * r_world_b[2] - ang_vel_b[2] * r_world_b[1];
            v_contact_b[1] = lin_vel_b[1] + ang_vel_b[2] * r_world_b[0] - ang_vel_b[0] * r_world_b[2];
            v_contact_b[2] = lin_vel_b[2] + ang_vel_b[0] * r_world_b[1] - ang_vel_b[1] * r_world_b[0];

            Real r_world_prev_b[3];
            cudaPhysics::quatRotateVector(r_world_prev_b, orient_prev_b, col.local_point_b);

            v_contact_prev_b[0] = lin_vel_prev_b[0] + ang_vel_prev_b[1] * r_world_prev_b[2] - ang_vel_prev_b[2] * r_world_prev_b[1];
            v_contact_prev_b[1] = lin_vel_prev_b[1] + ang_vel_prev_b[2] * r_world_prev_b[0] - ang_vel_prev_b[0] * r_world_prev_b[2];
            v_contact_prev_b[2] = lin_vel_prev_b[2] + ang_vel_prev_b[0] * r_world_prev_b[1] - ang_vel_prev_b[1] * r_world_prev_b[0];
        }

        Real rel_vel[3];
        cudaPhysics::vecSubs3(rel_vel, v_contact_a, v_contact_b);
        Real vn = cudaPhysics::dot3(rel_vel, col.normal);

        Real rel_vel_prev[3];
        cudaPhysics::vecSubs3(rel_vel_prev, v_contact_prev_a, v_contact_prev_b);
        Real vn_prev = cudaPhysics::dot3(rel_vel_prev, col.normal);

        Real restitution = data.m_restitution;
        Real constraint_vel_correction[3];
        cudaPhysics::vecMul3(constraint_vel_correction, -vn, col.normal);

        Real bounce = -restitution * vn_prev;
        if (bounce < 0)
            cudaPhysics::axpby(constraint_vel_correction, (Real)1.0, constraint_vel_correction, bounce, col.normal, 3);

        Real correction_len = cudaPhysics::len3(constraint_vel_correction);
        if (correction_len < static_cast<Real>(1e-10))
            return;

        Real n[3];
        cudaPhysics::vecMul3(n, static_cast<Real>(1.0) / correction_len, constraint_vel_correction);

        Real r_cross_n_a[3];
        cudaPhysics::cross3(r_cross_n_a, r_world_a, n);
        Real I_r_cross_n_a[3];
        cudaPhysics::matVec3(I_r_cross_n_a, inv_I_world_a, r_cross_n_a);
        Real w_a = inv_m_a + cudaPhysics::dot3(r_cross_n_a, I_r_cross_n_a);

        Real w_b = static_cast<Real>(0.0);
        Real I_r_cross_n_b[3] = {static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)};
        if (body_id_b >= 0)
        {
            Real r_cross_n_b[3];
            Real neg_n[3];
            cudaPhysics::vecMul3(neg_n, static_cast<Real>(-1.0), n);
            cudaPhysics::cross3(r_cross_n_b, r_world_b, neg_n);

            cudaPhysics::matVec3(I_r_cross_n_b, inv_I_world_b, r_cross_n_b);
            w_b = inv_m_b + cudaPhysics::dot3(r_cross_n_b, I_r_cross_n_b);
        }

        Real w_total = w_a + w_b;
        if (w_total < static_cast<Real>(1e-10))
            return;

        Real delta_p_a[3] = {static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)};
        Real delta_omega_a[3] = {static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)};
        Real delta_p_b[3] = {static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)};
        Real delta_omega_b[3] = {static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)};

        Real normal_impulse = correction_len / w_total;
        cudaPhysics::axpby(delta_p_a, (Real)1.0, delta_p_a, normal_impulse * inv_m_a, n, 3);
        cudaPhysics::axpby(delta_omega_a, (Real)1.0, delta_omega_a, normal_impulse, I_r_cross_n_a, 3);

        if (body_id_b >= 0)
        {
            cudaPhysics::axpby(delta_p_b, (Real)1.0, delta_p_b, -normal_impulse * inv_m_b, n, 3);
            cudaPhysics::axpby(delta_omega_b, (Real)1.0, delta_omega_b, normal_impulse, I_r_cross_n_b, 3);
        }

        Real vt[3];
        cudaPhysics::vecMul3(vt, vn, col.normal);
        Real tangent_vel[3];
        cudaPhysics::vecSubs3(tangent_vel, rel_vel, vt);

        Real tangent_speed = cudaPhysics::len3(tangent_vel);
        Real friction_coef = data.m_friction;
        Real max_friction_impulse = abs(normal_impulse) * friction_coef;
        Real friction_impulse = min(tangent_speed, max_friction_impulse);

        printf("body_id_a:%d, body_id_b:%d, vn:%f, vn_prev:%f, restitution:%f, bounce:%f, correction_len:%f, normal_impulse:%f, tangent_speed:%f, max_friction_impulse:%f, friction_impulse:%f\n",
            body_id_a, body_id_b, vn, vn_prev, restitution, bounce, correction_len, normal_impulse, tangent_speed, max_friction_impulse, friction_impulse);

        Real t[3];
        cudaPhysics::vecMul3(t, static_cast<Real>(1.0) / tangent_speed, tangent_vel);

        Real r_cross_t_a[3];
        cudaPhysics::cross3(r_cross_t_a, r_world_a, t);
        Real I_r_cross_t_a[3];
        cudaPhysics::matVec3(I_r_cross_t_a, inv_I_world_a, r_cross_t_a);
        Real w_t_a = inv_m_a + cudaPhysics::dot3(r_cross_t_a, I_r_cross_t_a);

        Real w_t_b = static_cast<Real>(0.0);
        Real I_r_cross_t_b[3] = {static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)};
        if (body_id_b >= 0)
        {
            Real r_cross_t_b[3];
            Real neg_t[3];
            cudaPhysics::vecMul3(neg_t, static_cast<Real>(-1.0), t);
            cudaPhysics::cross3(r_cross_t_b, r_world_b, neg_t);
            cudaPhysics::matVec3(I_r_cross_t_b, inv_I_world_b, r_cross_t_b);
            w_t_b = inv_m_b + cudaPhysics::dot3(r_cross_t_b, I_r_cross_t_b);
        }

        Real w_t_total = w_t_a + w_t_b;
        if (w_t_total >= static_cast<Real>(1e-10))
        {
            Real friction_factor = friction_impulse / w_t_total;
            cudaPhysics::axpby(delta_p_a, (Real)1.0, delta_p_a, -friction_factor * inv_m_a, t, 3);
            cudaPhysics::axpby(delta_omega_a, (Real)1.0, delta_omega_a, -friction_factor, I_r_cross_t_a, 3);

            if (body_id_b >= 0)
            {
                cudaPhysics::axpby(delta_p_b, (Real)1.0, delta_p_b, friction_factor * inv_m_b, t, 3);
                cudaPhysics::axpby(delta_omega_b, (Real)1.0, delta_omega_b, -friction_factor, I_r_cross_t_b, 3);
            }
        }

        atomicAdd(&data.dev_linear_velocity[body_id_a * 3 + 0], delta_p_a[0]);
        atomicAdd(&data.dev_linear_velocity[body_id_a * 3 + 1], delta_p_a[1]);
        atomicAdd(&data.dev_linear_velocity[body_id_a * 3 + 2], delta_p_a[2]);
        atomicAdd(&data.dev_angular_velocity[body_id_a * 3 + 0], delta_omega_a[0]);
        atomicAdd(&data.dev_angular_velocity[body_id_a * 3 + 1], delta_omega_a[1]);
        atomicAdd(&data.dev_angular_velocity[body_id_a * 3 + 2], delta_omega_a[2]);

        if (body_id_b >= 0)
        {
            atomicAdd(&data.dev_linear_velocity[body_id_b * 3 + 0], delta_p_b[0]);
            atomicAdd(&data.dev_linear_velocity[body_id_b * 3 + 1], delta_p_b[1]);
            atomicAdd(&data.dev_linear_velocity[body_id_b * 3 + 2], delta_p_b[2]);
            atomicAdd(&data.dev_angular_velocity[body_id_b * 3 + 0], delta_omega_b[0]);
            atomicAdd(&data.dev_angular_velocity[body_id_b * 3 + 1], delta_omega_b[1]);
            atomicAdd(&data.dev_angular_velocity[body_id_b * 3 + 2], delta_omega_b[2]);
        }
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
    : m_ground_collision(static_cast<Real>(0.0)),
      m_body_collision()
{
    m_data.num_bodies = static_cast<unsigned int>(mass.size());
    m_max_collisions = (static_cast<unsigned int>(mass.size()) + static_cast<unsigned int>(mass.size() * (mass.size() - 1) / 2)) * MAX_MANIFOLD_POINTS;

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
    cudaMalloc(&m_data.dev_linear_velocity_before_constraint, sizeof(Real) * m_data.num_bodies * 3);
    cudaMalloc(&m_data.dev_angular_velocity_before_constraint, sizeof(Real) * m_data.num_bodies * 3);

    cudaMalloc(&m_data.dev_delta_position, sizeof(Real) * m_data.num_bodies * 3);
    cudaMalloc(&m_data.dev_delta_omega, sizeof(Real) * m_data.num_bodies * 3);
    cudaMalloc(&m_data.dev_constraint_inv_weight, sizeof(Real) * m_data.num_bodies);

    cudaMalloc(&m_collision_data.dev_collisions, sizeof(CollisionInfo<Real>) * m_max_collisions);
    cudaMalloc(&m_collision_data.dev_collision_count, sizeof(int));
    m_collision_data.max_collisions = m_max_collisions;

    RigidSolverKernel::compute_inertia_tensor<Real><<<CUDA_GRID_SIZE(m_data.num_bodies), CUDA_BLOCK_SIZE>>>(m_data);

    m_data.m_time_step = static_cast<Real>(0.002);
    m_data.m_gravity[0] = static_cast<Real>(0.0);
    m_data.m_gravity[1] = static_cast<Real>(-9.8);
    m_data.m_gravity[2] = static_cast<Real>(0.0);
    m_data.m_damping = static_cast<Real>(0.9999);
    m_data.m_restitution = static_cast<Real>(0.3);
    m_data.m_friction = static_cast<Real>(0.5);
    m_data.m_num_substeps = 1;
    m_data.m_num_solver_iterations = 10;
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
    if (m_data.dev_linear_velocity_before_constraint)
        cudaFree(m_data.dev_linear_velocity_before_constraint);
    if (m_data.dev_angular_velocity_before_constraint)
        cudaFree(m_data.dev_angular_velocity_before_constraint);
    if (m_data.dev_delta_position)
        cudaFree(m_data.dev_delta_position);
    if (m_data.dev_delta_omega)
        cudaFree(m_data.dev_delta_omega);
    if (m_data.dev_constraint_inv_weight)
        cudaFree(m_data.dev_constraint_inv_weight);
    if (m_collision_data.dev_collisions)
        cudaFree(m_collision_data.dev_collisions);
    if (m_collision_data.dev_collision_count)
        cudaFree(m_collision_data.dev_collision_count);
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

        cudaMemset(m_collision_data.dev_collision_count, 0, sizeof(int));
        m_ground_collision.Detect(
            &m_collision_data,
            m_max_collisions,
            m_data.num_bodies,
            m_data.dev_position,
            m_data.dev_orientation,
            m_data.dev_shape,
            m_data.dev_shape_param);
        m_body_collision.Detect(
            &m_collision_data,
            m_max_collisions,
            m_data.num_bodies,
            m_data.dev_position,
            m_data.dev_orientation,
            m_data.dev_shape,
            m_data.dev_shape_param);

        for (unsigned int iter = 0; iter < m_data.m_num_solver_iterations; ++iter)
        {
            cudaMemset(m_data.dev_delta_position, 0, sizeof(Real) * m_data.num_bodies * 3);
            cudaMemset(m_data.dev_delta_omega, 0, sizeof(Real) * m_data.num_bodies * 3);
            cudaMemset(m_data.dev_constraint_inv_weight, 0, sizeof(Real) * m_data.num_bodies);
            RigidSolverKernel::accumulate_collision_deltas<Real><<<CUDA_GRID_SIZE(m_max_collisions), CUDA_BLOCK_SIZE>>>(
                m_data, m_collision_data.dev_collisions, m_collision_data.dev_collision_count);
            RigidSolverKernel::apply_deltas<Real><<<CUDA_GRID_SIZE(m_data.num_bodies), CUDA_BLOCK_SIZE>>>(m_data);
            RigidSolverKernel::compute_inv_inertia_world<Real><<<CUDA_GRID_SIZE(m_data.num_bodies), CUDA_BLOCK_SIZE>>>(m_data);
        }

        RigidSolverKernel::update_velocity<Real><<<CUDA_GRID_SIZE(m_data.num_bodies), CUDA_BLOCK_SIZE>>>(m_data);

        RigidSolverKernel::solve_contact_velocities<Real><<<CUDA_GRID_SIZE(m_max_collisions), CUDA_BLOCK_SIZE>>>(
            m_data, m_collision_data.dev_collisions, m_collision_data.dev_collision_count);

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

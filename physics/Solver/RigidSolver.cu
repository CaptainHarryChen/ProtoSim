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

        cudaPhysics::vecCopy3(pos_prev, pos);
        cudaPhysics::vecCopy(orient_prev, orient, 4);

        cudaPhysics::axpby(lin_vel, static_cast<Real>(1.0), lin_vel, data.m_time_step, data.m_gravity, 3);
        cudaPhysics::vecMul3(lin_vel, data.m_damping, lin_vel);
        cudaPhysics::vecMul3(ang_vel, data.m_damping, ang_vel);

        cudaPhysics::axpby(pos, static_cast<Real>(1.0), pos, data.m_time_step, lin_vel, 3);
        cudaPhysics::quatAddAngularVelocity(orient, ang_vel, data.m_time_step);
    }

    template <typename Real>
    __global__ void solve_ground_collision(
        RigidSolverData<Real> data,
        const CollisionInfo<Real>* collisions)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data.num_bodies)
            return;

        const CollisionInfo<Real>& col = collisions[i];
        if (col.body_id_a < 0)
            return;

        Real *pos = &data.dev_position[i * 3];
        Real *orient = &data.dev_orientation[i * 4];
        Real *lin_vel = &data.dev_linear_velocity[i * 3];
        Real *ang_vel = &data.dev_angular_velocity[i * 3];

        Real r_world[3];
        cudaPhysics::quatRotateVector(r_world, orient, col.local_point_a);

        Real inv_m = data.dev_inv_mass[i];
        Real *inv_I_local = &data.dev_inv_inertia_tensor_local[i * 9];

        Real R[9];
        cudaPhysics::quatToRotationMatrix(R, orient);

        Real inv_I_world[9], RT[9], temp[9];
        cudaPhysics::matTrans3(RT, R);
        cudaPhysics::matMul3(temp, RT, inv_I_local);
        cudaPhysics::matMul3(inv_I_world, temp, R);

        Real r_cross_n[3];
        cudaPhysics::cross3(r_cross_n, r_world, col.normal);

        Real w = inv_m;
        Real I_r_cross_n[3];
        cudaPhysics::matVec3(I_r_cross_n, inv_I_world, r_cross_n);
        w += cudaPhysics::dot3(r_cross_n, I_r_cross_n);

        Real lambda = col.penetration / w;

        pos[1] += lambda * inv_m;

        cudaPhysics::axpby(ang_vel, static_cast<Real>(1.0), ang_vel, lambda, I_r_cross_n, 3);

        Real v_contact[3];
        v_contact[0] = lin_vel[0] + ang_vel[1] * r_world[2] - ang_vel[2] * r_world[1];
        v_contact[1] = lin_vel[1] + ang_vel[2] * r_world[0] - ang_vel[0] * r_world[2];
        v_contact[2] = lin_vel[2] + ang_vel[0] * r_world[1] - ang_vel[1] * r_world[0];

        Real v_n = cudaPhysics::dot3(v_contact, col.normal);
        if (v_n < 0)
        {
            Real v_n_new = -data.m_restitution * v_n;
            Real delta_v_n = v_n_new - v_n;
            Real lambda_n = delta_v_n / w;

            lin_vel[1] += lambda_n * inv_m;
            cudaPhysics::axpby(ang_vel, static_cast<Real>(1.0), ang_vel, lambda_n, I_r_cross_n, 3);
        }
    }

    template <typename Real>
    __global__ void apply_friction(
        RigidSolverData<Real> data,
        const CollisionInfo<Real>* collisions)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= data.num_bodies)
            return;

        const CollisionInfo<Real>& col = collisions[i];
        if (col.body_id_a < 0)
            return;

        Real *pos = &data.dev_position[i * 3];
        Real *orient = &data.dev_orientation[i * 4];
        Real *lin_vel = &data.dev_linear_velocity[i * 3];
        Real *ang_vel = &data.dev_angular_velocity[i * 3];

        Real r_world[3];
        cudaPhysics::quatRotateVector(r_world, orient, col.local_point_a);

        Real world_point[3];
        cudaPhysics::vecAdd3(world_point, pos, r_world);

        Real ground_y = col.normal[1] > 0 ? pos[1] - r_world[1] + col.penetration : pos[1] - r_world[1];
        if (world_point[1] > ground_y + static_cast<Real>(0.001))
            return;

        Real v_contact[3];
        v_contact[0] = lin_vel[0] + ang_vel[1] * r_world[2] - ang_vel[2] * r_world[1];
        v_contact[1] = lin_vel[1] + ang_vel[2] * r_world[0] - ang_vel[0] * r_world[2];
        v_contact[2] = lin_vel[2] + ang_vel[0] * r_world[1] - ang_vel[1] * r_world[0];

        Real v_n = cudaPhysics::dot3(v_contact, col.normal);
        Real v_t[3];
        v_t[0] = v_contact[0] - v_n * col.normal[0];
        v_t[1] = v_contact[1] - v_n * col.normal[1];
        v_t[2] = v_contact[2] - v_n * col.normal[2];

        Real v_t_len = cudaPhysics::len3(v_t);
        if (v_t_len < static_cast<Real>(1e-6))
            return;

        Real inv_m = data.dev_inv_mass[i];
        Real *inv_I_local = &data.dev_inv_inertia_tensor_local[i * 9];

        Real R[9];
        cudaPhysics::quatToRotationMatrix(R, orient);

        Real inv_I_world[9], RT[9], temp[9];
        cudaPhysics::matTrans3(RT, R);
        cudaPhysics::matMul3(temp, RT, inv_I_local);
        cudaPhysics::matMul3(inv_I_world, temp, R);

        Real t[3];
        cudaPhysics::vecMul3(t, static_cast<Real>(1.0) / v_t_len, v_t);

        Real r_cross_t[3];
        cudaPhysics::cross3(r_cross_t, r_world, t);

        Real w_t = inv_m;
        Real I_r_cross_t[3];
        cudaPhysics::matVec3(I_r_cross_t, inv_I_world, r_cross_t);
        w_t += cudaPhysics::dot3(r_cross_t, I_r_cross_t);

        Real lambda_t = v_t_len / w_t;
        Real max_friction = data.m_friction * abs(v_n) / w_t;
        if (lambda_t > max_friction)
            lambda_t = max_friction;

        Real delta_lin[3];
        cudaPhysics::vecMul3(delta_lin, -lambda_t * inv_m, t);
        cudaPhysics::vecAdd3(lin_vel, lin_vel, delta_lin);

        Real delta_ang[3];
        cudaPhysics::vecMul3(delta_ang, -lambda_t, I_r_cross_t);
        cudaPhysics::vecAdd3(ang_vel, ang_vel, delta_ang);
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
    {
        m_data.dev_position = nullptr;
        m_data.dev_orientation = nullptr;
        m_data.dev_linear_velocity = nullptr;
        m_data.dev_angular_velocity = nullptr;
        m_data.dev_mass = nullptr;
        m_data.dev_inv_mass = nullptr;
        m_data.dev_inertia_tensor_local = nullptr;
        m_data.dev_inv_inertia_tensor_local = nullptr;
        m_data.dev_shape = nullptr;
        m_data.dev_shape_param = nullptr;
        m_data.dev_position_prev = nullptr;
        m_data.dev_orientation_prev = nullptr;
        return;
    }

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

    cudaMalloc(&m_data.dev_shape, sizeof(int) * m_data.num_bodies);
    cudaMemcpy(m_data.dev_shape, shape.data(), sizeof(int) * m_data.num_bodies, cudaMemcpyHostToDevice);

    cudaMalloc(&m_data.dev_shape_param, sizeof(Real) * m_data.num_bodies * 3);
    cudaMemcpy(m_data.dev_shape_param, shape_param.data(), sizeof(Real) * m_data.num_bodies * 3, cudaMemcpyHostToDevice);

    cudaMalloc(&m_data.dev_position_prev, sizeof(Real) * m_data.num_bodies * 3);
    cudaMalloc(&m_data.dev_orientation_prev, sizeof(Real) * m_data.num_bodies * 4);

    RigidSolverKernel::compute_inertia_tensor<Real><<<CUDA_GRID_SIZE(m_data.num_bodies), CUDA_BLOCK_SIZE>>>(m_data);

    m_data.m_time_step = static_cast<Real>(0.016);
    m_data.m_gravity[0] = static_cast<Real>(0.0);
    m_data.m_gravity[1] = static_cast<Real>(-9.8);
    m_data.m_gravity[2] = static_cast<Real>(0.0);
    m_data.m_damping = static_cast<Real>(0.999);
    m_data.m_ground_height = static_cast<Real>(0.0);
    m_data.m_restitution = static_cast<Real>(0.3);
    m_data.m_friction = static_cast<Real>(0.5);
    m_data.m_num_substeps = 4;
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
    if (m_data.dev_shape)
        cudaFree(m_data.dev_shape);
    if (m_data.dev_shape_param)
        cudaFree(m_data.dev_shape_param);
    if (m_data.dev_position_prev)
        cudaFree(m_data.dev_position_prev);
    if (m_data.dev_orientation_prev)
        cudaFree(m_data.dev_orientation_prev);
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

        m_ground_collision.Detect(
            m_data.dev_position,
            m_data.dev_orientation,
            m_data.dev_shape,
            m_data.dev_shape_param);

        for (unsigned int iter = 0; iter < 1; ++iter)
        {
            RigidSolverKernel::solve_ground_collision<Real><<<CUDA_GRID_SIZE(m_data.num_bodies), CUDA_BLOCK_SIZE>>>(
                m_data, m_ground_collision.GetData().dev_collisions);
        }

        RigidSolverKernel::apply_friction<Real><<<CUDA_GRID_SIZE(m_data.num_bodies), CUDA_BLOCK_SIZE>>>(
            m_data, m_ground_collision.GetData().dev_collisions);

        RigidSolverKernel::update_velocity<Real><<<CUDA_GRID_SIZE(m_data.num_bodies), CUDA_BLOCK_SIZE>>>(m_data);

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

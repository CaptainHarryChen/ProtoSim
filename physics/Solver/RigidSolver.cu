#include "RigidSolver.cuh"
#include <cuda_utils/cuda_utils.cuh>

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
            Real hx = p0;
            Real hy = p1;
            Real hz = p2;
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
            Real r = p0;
            Real hh = p1;
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
        I[6] = 0;   I[7] = 0; I[8] = Izz;

        Real *invI = &data.dev_inv_inertia_tensor_local[i * 9];
        invI[0] = static_cast<Real>(1.0) / Ixx; invI[1] = 0; invI[2] = 0;
        invI[3] = 0;   invI[4] = static_cast<Real>(1.0) / Iyy; invI[5] = 0;
        invI[6] = 0;   invI[7] = 0; invI[8] = static_cast<Real>(1.0) / Izz;
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

    RigidSolverKernel::compute_inertia_tensor<Real><<<CUDA_GRID_SIZE(m_data.num_bodies), CUDA_BLOCK_SIZE>>>(m_data);
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
}

template <typename Real>
void RigidSolver<Real>::Step()
{
}

template <typename Real>
Real *RigidSolver<Real>::GetDevicePositions()
{
    return m_data.dev_position;
}

template struct RigidSolverData<float>;
template struct RigidSolverData<double>;
template class RigidSolver<float>;
template class RigidSolver<double>;

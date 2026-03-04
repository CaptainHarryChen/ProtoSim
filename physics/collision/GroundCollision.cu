#include "GroundCollision.cuh"
#include <cuda_utils/cuda_utils.cuh>
#include <Math/algebra.cuh>

namespace GroundCollisionKernel
{
    template <typename Real>
    __device__ void get_lowest_point_sphere(Real *lowest, const Real *pos, const Real *orient, Real radius)
    {
        Real down[3] = {0, -1, 0};
        Real rotated_down[3];
        cudaPhysics::quatRotateVector(rotated_down, orient, down);
        cudaPhysics::axpby(lowest, static_cast<Real>(1.0), pos, radius, rotated_down, 3);
    }

    template <typename Real>
    __device__ void get_lowest_point_box(Real *lowest, const Real *pos, const Real *orient, Real hx, Real hy, Real hz)
    {
        Real corners[8][3] = {
            {-hx, -hy, -hz}, {hx, -hy, -hz}, {-hx, hy, -hz}, {hx, hy, -hz},
            {-hx, -hy, hz}, {hx, -hy, hz}, {-hx, hy, hz}, {hx, hy, hz}};

        Real min_y = 1e30;
        Real world_corner[3];

        for (int c = 0; c < 8; ++c)
        {
            Real rotated_corner[3];
            cudaPhysics::quatRotateVector(rotated_corner, orient, corners[c]);
            cudaPhysics::vecAdd3(world_corner, rotated_corner, pos);

            if (world_corner[1] < min_y)
            {
                min_y = world_corner[1];
                cudaPhysics::vecCopy3(lowest, world_corner);
            }
        }
    }

    template <typename Real>
    __device__ void get_lowest_point_capsule(Real *lowest, const Real *pos, const Real *orient, Real radius, Real half_height)
    {
        Real axis[3] = {0, 1, 0};
        Real rotated_axis[3];
        cudaPhysics::quatRotateVector(rotated_axis, orient, axis);

        Real top_center[3], bottom_center[3];
        cudaPhysics::axpby(top_center, static_cast<Real>(1.0), pos, half_height, rotated_axis, 3);
        cudaPhysics::axpby(bottom_center, static_cast<Real>(1.0), pos, -half_height, rotated_axis, 3);

        Real down[3] = {0, -1, 0};
        Real rotated_down[3];
        cudaPhysics::quatRotateVector(rotated_down, orient, down);

        Real top_lowest[3], bottom_lowest[3];
        cudaPhysics::axpby(top_lowest, static_cast<Real>(1.0), top_center, radius, rotated_down, 3);
        cudaPhysics::axpby(bottom_lowest, static_cast<Real>(1.0), bottom_center, radius, rotated_down, 3);

        if (top_lowest[1] < bottom_lowest[1])
            cudaPhysics::vecCopy3(lowest, top_lowest);
        else
            cudaPhysics::vecCopy3(lowest, bottom_lowest);
    }

    template <typename Real>
    __global__ void detect_ground_collision_kernel(
        CollisionInfo<Real>* collisions,
        unsigned int num_bodies,
        Real ground_height,
        const Real* position,
        const Real* orientation,
        const int* shape,
        const Real* shape_param)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= num_bodies)
            return;

        const Real* pos = &position[i * 3];
        const Real* orient = &orientation[i * 4];
        int s = shape[i];
        Real p0 = shape_param[i * 3 + 0];
        Real p1 = shape_param[i * 3 + 1];
        Real p2 = shape_param[i * 3 + 2];

        Real lowest[3];
        if (s == RIGID_BODY_SPHERE)
            get_lowest_point_sphere(lowest, pos, orient, p0);
        else if (s == RIGID_BODY_BOX)
            get_lowest_point_box(lowest, pos, orient, p0, p1, p2);
        else
            get_lowest_point_capsule(lowest, pos, orient, p0, p1);

        Real penetration = ground_height - lowest[1];
        if (penetration <= 0)
        {
            collisions[i].body_id_a = -1;
            collisions[i].body_id_b = -1;
            return;
        }

        Real r[3];
        cudaPhysics::vecSubs3(r, lowest, pos);

        Real orient_conj[4];
        cudaPhysics::quatConjugate(orient_conj, orient);
        Real local_point[3];
        cudaPhysics::quatRotateVector(local_point, orient_conj, r);

        collisions[i].body_id_a = static_cast<int>(i);
        collisions[i].body_id_b = -1;
        collisions[i].local_point_a[0] = local_point[0];
        collisions[i].local_point_a[1] = local_point[1];
        collisions[i].local_point_a[2] = local_point[2];
        collisions[i].normal[0] = static_cast<Real>(0.0);
        collisions[i].normal[1] = static_cast<Real>(1.0);
        collisions[i].normal[2] = static_cast<Real>(0.0);
        collisions[i].penetration = penetration;
    }
}

template <typename Real>
GroundCollision<Real>::GroundCollision(unsigned int num_bodies, Real ground_height)
{
    m_data.max_collisions = num_bodies;
    m_num_bodies = num_bodies;
    m_ground_height = ground_height;

    if (num_bodies == 0)
    {
        m_data.dev_collisions = nullptr;
        m_data.dev_collision_count = nullptr;
        return;
    }

    cudaMalloc(&m_data.dev_collisions, sizeof(CollisionInfo<Real>) * num_bodies);
    cudaMalloc(&m_data.dev_collision_count, sizeof(int));
}

template <typename Real>
GroundCollision<Real>::~GroundCollision()
{
    if (m_data.dev_collisions)
        cudaFree(m_data.dev_collisions);
    if (m_data.dev_collision_count)
        cudaFree(m_data.dev_collision_count);
}

template <typename Real>
void GroundCollision<Real>::Detect(
    const Real* dev_position,
    const Real* dev_orientation,
    const int* dev_shape,
    const Real* dev_shape_param)
{
    if (m_num_bodies == 0)
        return;

    GroundCollisionKernel::detect_ground_collision_kernel<Real><<<CUDA_GRID_SIZE(m_num_bodies), CUDA_BLOCK_SIZE>>>(
        m_data.dev_collisions,
        m_num_bodies,
        m_ground_height,
        dev_position,
        dev_orientation,
        dev_shape,
        dev_shape_param);
    cudaMemcpy(m_data.dev_collision_count, &m_num_bodies, sizeof(int), cudaMemcpyHostToDevice);
}

template class GroundCollision<float>;
template class GroundCollision<double>;

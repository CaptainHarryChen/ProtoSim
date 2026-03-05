#include "GroundCollision.cuh"
#include <cuda_utils/cuda_utils.cuh>
#include <Math/algebra.cuh>

#define EPSILON 1e-6

namespace GroundCollisionKernel
{
    template <typename Real>
    __device__ void get_lowest_point_sphere(Real *lowest, const Real *pos, Real radius)
    {
        cudaPhysics::vecCopy3(lowest, pos);
        lowest[1] -= radius;
    }

    template <typename Real>
    __device__ void get_lowest_point_box(Real *lowest, const Real *pos, const Real *orient, Real hx, Real hy, Real hz)
    {
        Real corners[8][3] = {
            {-hx, -hy, -hz}, {hx, -hy, -hz}, {-hx, hy, -hz}, {hx, hy, -hz}, {-hx, -hy, hz}, {hx, -hy, hz}, {-hx, hy, hz}, {hx, hy, hz}};

        Real min_y = 1e30;
        Real sum[3] = {0.0, 0.0, 0.0}, num_point = 0.0;
        Real world_corner[3];

        for (int c = 0; c < 8; ++c)
        {
            Real rotated_corner[3];
            cudaPhysics::quatRotateVector(rotated_corner, orient, corners[c]);
            cudaPhysics::vecAdd3(world_corner, rotated_corner, pos);

            if (abs(world_corner[1] - min_y) < EPSILON)
            {
                cudaPhysics::vecAdd3(sum, world_corner, sum);
                num_point += 1.0;
            }
            else if (world_corner[1] < min_y)
            {
                min_y = world_corner[1];
                cudaPhysics::vecCopy3(sum, world_corner);
                num_point = 1.0;
            }
        }
        cudaPhysics::vecMul3(lowest, (Real)(1.0 / num_point), sum);
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
        Real lowest1[3], lowest2[3];
        get_lowest_point_sphere(lowest1, top_center, radius);
        get_lowest_point_sphere(lowest2, bottom_center, radius);
        if (abs(lowest1[1] - lowest2[1]) < EPSILON)
        {
            cudaPhysics::vecCopy3(lowest, pos);
            lowest[1] -= radius;
        }
        else if (lowest1[1] < lowest2[1])
            cudaPhysics::vecCopy3(lowest, lowest1);
        else
            cudaPhysics::vecCopy3(lowest, lowest2);
    }

    template <typename Real>
    __global__ void detect_ground_collision_kernel(
        CollisionInfo<Real> *collisions,
        int *collision_count,
        unsigned int max_collisions,
        unsigned int num_bodies,
        Real ground_height,
        const Real *position,
        const Real *orientation,
        const int *shape,
        const Real *shape_param)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= num_bodies)
            return;

        const Real *pos = &position[i * 3];
        const Real *orient = &orientation[i * 4];
        int s = shape[i];
        Real p0 = shape_param[i * 3 + 0];
        Real p1 = shape_param[i * 3 + 1];
        Real p2 = shape_param[i * 3 + 2];

        Real lowest[3];
        if (s == RIGID_BODY_SPHERE)
            get_lowest_point_sphere(lowest, pos, p0);
        else if (s == RIGID_BODY_BOX)
            get_lowest_point_box(lowest, pos, orient, p0, p1, p2);
        else
            get_lowest_point_capsule(lowest, pos, orient, p0, p1);

        Real penetration = ground_height - lowest[1];
        if (penetration <= 0)
            return;

        int slot = atomicAdd(collision_count, 1);
        if (slot >= static_cast<int>(max_collisions))
            return;

        Real r[3];
        cudaPhysics::vecSubs3(r, lowest, pos);

        Real orient_conj[4];
        cudaPhysics::quatConjugate(orient_conj, orient);
        Real local_point[3];
        cudaPhysics::quatRotateVector(local_point, orient_conj, r);

        collisions[slot].body_id_a = static_cast<int>(i);
        collisions[slot].body_id_b = -1;
        collisions[slot].local_point_a[0] = local_point[0];
        collisions[slot].local_point_a[1] = local_point[1];
        collisions[slot].local_point_a[2] = local_point[2];
        collisions[slot].local_point_b[0] = lowest[0];
        collisions[slot].local_point_b[1] = ground_height;
        collisions[slot].local_point_b[2] = lowest[2];
        collisions[slot].normal[0] = static_cast<Real>(0.0);
        collisions[slot].normal[1] = static_cast<Real>(-1.0);
        collisions[slot].normal[2] = static_cast<Real>(0.0);
    }
}

template <typename Real>
GroundCollision<Real>::GroundCollision(Real ground_height)
{
    m_ground_height = ground_height;
}

template <typename Real>
GroundCollision<Real>::~GroundCollision()
{
}

template <typename Real>
void GroundCollision<Real>::Detect(
    CollisionData<Real> *collision_data,
    unsigned int max_collisions,
    unsigned int num_bodies,
    const Real *dev_position,
    const Real *dev_orientation,
    const int *dev_shape,
    const Real *dev_shape_param)
{
    GroundCollisionKernel::detect_ground_collision_kernel<Real><<<CUDA_GRID_SIZE(num_bodies), CUDA_BLOCK_SIZE>>>(
        collision_data->dev_collisions,
        collision_data->dev_collision_count,
        num_bodies,
        max_collisions,
        m_ground_height,
        dev_position,
        dev_orientation,
        dev_shape,
        dev_shape_param);
}

template class GroundCollision<float>;
template class GroundCollision<double>;

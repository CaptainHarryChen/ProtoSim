#include "GroundCollision.cuh"
#include <cuda_utils/cuda_utils.cuh>
#include <Math/algebra.cuh>
#include "CollisionDataUtils.cuh"

namespace GroundCollisionKernel
{
    template <typename Real>
    __device__ int add_collision_point(
        CollisionInfo<Real> *collisions,
        int *collision_count,
        unsigned int max_collisions,
        int body_id,
        const Real *pos,
        const Real *orient,
        const Real *world_point,
        Real ground_height)
    {
        int slot = atomicAdd(collision_count, 1);
        if (slot >= static_cast<int>(max_collisions))
            return 0;

        Real orient_conj[4];
        cudaPhysics::quatConjugate(orient_conj, orient);

        Real local_point[3];
        cudaPhysics::get_local_point(local_point, pos, orient_conj, world_point);

        collisions[slot].body_id_a = body_id;
        collisions[slot].body_id_b = -1;
        collisions[slot].local_point_a[0] = local_point[0];
        collisions[slot].local_point_a[1] = local_point[1];
        collisions[slot].local_point_a[2] = local_point[2];
        collisions[slot].local_point_b[0] = world_point[0];
        collisions[slot].local_point_b[1] = ground_height;
        collisions[slot].local_point_b[2] = world_point[2];
        collisions[slot].normal[0] = static_cast<Real>(0.0);
        collisions[slot].normal[1] = static_cast<Real>(-1.0);
        collisions[slot].normal[2] = static_cast<Real>(0.0);

        return 1;
    }

    template <typename Real>
    __device__ int detect_sphere_ground(
        CollisionInfo<Real> *collisions,
        int *collision_count,
        unsigned int max_collisions,
        int body_id,
        const Real *pos,
        const Real *orient,
        Real radius,
        Real ground_height)
    {
        Real lowest_y = pos[1] - radius;
        Real penetration = ground_height - lowest_y;

        if (penetration <= static_cast<Real>(0.0))
            return 0;

        Real lowest[3];
        lowest[0] = pos[0];
        lowest[1] = lowest_y;
        lowest[2] = pos[2];

        return add_collision_point(collisions, collision_count, max_collisions, body_id, pos, orient, lowest, ground_height);
    }

    template <typename Real>
    __device__ int detect_capsule_ground(
        CollisionInfo<Real> *collisions,
        int *collision_count,
        unsigned int max_collisions,
        int body_id,
        const Real *pos,
        const Real *orient,
        Real radius,
        Real half_height,
        Real ground_height)
    {
        Real axis[3] = {static_cast<Real>(0.0), static_cast<Real>(1.0), static_cast<Real>(0.0)};
        Real rotated_axis[3];
        cudaPhysics::quatRotateVector(rotated_axis, orient, axis);

        Real top_center[3], bottom_center[3];
        cudaPhysics::axpby(top_center, static_cast<Real>(1.0), pos, half_height, rotated_axis, 3);
        cudaPhysics::axpby(bottom_center, static_cast<Real>(1.0), pos, -half_height, rotated_axis, 3);

        Real top_lowest[3], bottom_lowest[3];
        top_lowest[0] = top_center[0];
        top_lowest[1] = top_center[1] - radius;
        top_lowest[2] = top_center[2];

        bottom_lowest[0] = bottom_center[0];
        bottom_lowest[1] = bottom_center[1] - radius;
        bottom_lowest[2] = bottom_center[2];

        int num_contacts = 0;

        Real pen_top = ground_height - top_lowest[1];
        if (pen_top > static_cast<Real>(0.0))
        {
            num_contacts += add_collision_point(collisions, collision_count, max_collisions, body_id, pos, orient, top_lowest, ground_height);
        }

        Real pen_bottom = ground_height - bottom_lowest[1];
        if (pen_bottom > static_cast<Real>(0.0))
        {
            num_contacts += add_collision_point(collisions, collision_count, max_collisions, body_id, pos, orient, bottom_lowest, ground_height);
        }

        return num_contacts;
    }

    template <typename Real>
    __device__ int detect_box_ground(
        CollisionInfo<Real> *collisions,
        int *collision_count,
        unsigned int max_collisions,
        int body_id,
        const Real *pos,
        const Real *orient,
        Real hx, Real hy, Real hz,
        Real ground_height,
        Real epsilon)
    {
        Real corners[8][3] = {
            {-hx, -hy, -hz}, {hx, -hy, -hz}, {-hx, hy, -hz}, {hx, hy, -hz}, {-hx, -hy, hz}, {hx, -hy, hz}, {-hx, hy, hz}, {hx, hy, hz}};
        Real world_corners[8][3];

        for (int c = 0; c < 8; ++c)
        {
            Real rotated_corner[3];
            cudaPhysics::quatRotateVector(rotated_corner, orient, corners[c]);
            cudaPhysics::vecAdd3(world_corners[c], rotated_corner, pos);
        }

        int num_contacts = 0;
        for (int i = 0; i < 8 && num_contacts < MAX_MANIFOLD_POINTS; ++i)
            if (world_corners[i][1] < ground_height)
                num_contacts += add_collision_point(collisions, collision_count, max_collisions, body_id, pos, orient, world_corners[i], ground_height);

        return num_contacts;
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

        Real epsilon = 1e-6;
        
        const Real *pos = &position[i * 3];
        const Real *orient = &orientation[i * 4];
        int s = shape[i];
        Real p0 = shape_param[i * 3 + 0];
        Real p1 = shape_param[i * 3 + 1];
        Real p2 = shape_param[i * 3 + 2];

        if (s == RIGID_BODY_SPHERE)
        {
            detect_sphere_ground(collisions, collision_count, max_collisions, static_cast<int>(i), pos, orient, p0, ground_height);
        }
        else if (s == RIGID_BODY_BOX)
        {
            detect_box_ground(collisions, collision_count, max_collisions, static_cast<int>(i), pos, orient, p0, p1, p2, ground_height, epsilon);
        }
        else
        {
            detect_capsule_ground(collisions, collision_count, max_collisions, static_cast<int>(i), pos, orient, p0, p1, ground_height);
        }
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
        max_collisions,
        num_bodies,
        m_ground_height,
        dev_position,
        dev_orientation,
        dev_shape,
        dev_shape_param);
}

template class GroundCollision<float>;
template class GroundCollision<double>;

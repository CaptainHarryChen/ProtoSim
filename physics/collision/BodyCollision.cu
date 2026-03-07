#include "BodyCollision.cuh"
#include <cuda_utils/cuda_utils.cuh>
#include <Math/algebra.cuh>
#include <collision/primitive/Capsule.cuh>
#include <collision/primitive/Sphere.cuh>
#include <collision/primitive/Box.cuh>

namespace BodyCollisionKernel
{
    template <typename Real>
    __device__ int detect_collision(
        CollisionInfo<Real> *collisions,
        int *collision_count,
        unsigned int max_collisions,
        int body_id_a, int body_id_b,
        int shape_a, const Real *pos_a, const Real *orient_a, const Real *params_a,
        int shape_b, const Real *pos_b, const Real *orient_b, const Real *params_b)
    {
        Real epsilon = 1e-6;

        if (shape_a == RIGID_BODY_SPHERE && shape_b == RIGID_BODY_SPHERE)
        {
            return collide_sphere_sphere(collisions, collision_count, max_collisions, body_id_a, body_id_b,
                                         pos_a, orient_a, params_a[0],
                                         pos_b, orient_b, params_b[0],
                                         epsilon);
        }
        else if (shape_a == RIGID_BODY_SPHERE && shape_b == RIGID_BODY_BOX)
        {
            return collide_sphere_box(collisions, collision_count, max_collisions, body_id_a, body_id_b,
                                      pos_a, orient_a, params_a[0],
                                      pos_b, orient_b, params_b[0], params_b[1], params_b[2], epsilon, false);
        }
        else if (shape_a == RIGID_BODY_BOX && shape_b == RIGID_BODY_SPHERE)
        {
            return collide_sphere_box(collisions, collision_count, max_collisions, body_id_b, body_id_a,
                                      pos_b, orient_b, params_b[0],
                                      pos_a, orient_a, params_a[0], params_a[1], params_a[2], epsilon, true);
        }
        else if (shape_a == RIGID_BODY_BOX && shape_b == RIGID_BODY_BOX)
        {
            return collide_box_box(collisions, collision_count, max_collisions, body_id_a, body_id_b,
                                   pos_a, orient_a, params_a[0], params_a[1], params_a[2],
                                   pos_b, orient_b, params_b[0], params_b[1], params_b[2], epsilon);
        }
        else if (shape_a == RIGID_BODY_CAPSULE && shape_b == RIGID_BODY_SPHERE)
        {
            return collide_capsule_sphere(collisions, collision_count, max_collisions, body_id_a, body_id_b,
                                          pos_a, orient_a, params_a[0], params_a[1],
                                          pos_b, orient_b, params_b[0], epsilon, false);
        }
        else if (shape_a == RIGID_BODY_SPHERE && shape_b == RIGID_BODY_CAPSULE)
        {
            return collide_capsule_sphere(collisions, collision_count, max_collisions, body_id_b, body_id_a,
                                          pos_b, orient_b, params_b[0], params_b[1],
                                          pos_a, orient_a, params_a[0], epsilon, true);
        }
        else if (shape_a == RIGID_BODY_CAPSULE && shape_b == RIGID_BODY_BOX)
        {
            return collide_capsule_box(collisions, collision_count, max_collisions, body_id_a, body_id_b,
                                       pos_a, orient_a, params_a[0], params_a[1],
                                       pos_b, orient_b, params_b[0], params_b[1], params_b[2], epsilon, false);
        }
        else if (shape_a == RIGID_BODY_BOX && shape_b == RIGID_BODY_CAPSULE)
        {
            return collide_capsule_box(collisions, collision_count, max_collisions, body_id_b, body_id_a,
                                       pos_b, orient_b, params_b[0], params_b[1],
                                       pos_a, orient_a, params_a[0], params_a[1], params_a[2], epsilon, true);
        }
        else if (shape_a == RIGID_BODY_CAPSULE && shape_b == RIGID_BODY_CAPSULE)
        {
            return collide_capsule_capsule(collisions, collision_count, max_collisions, body_id_a, body_id_b,
                                           pos_a, orient_a, params_a[0], params_a[1],
                                           pos_b, orient_b, params_b[0], params_b[1], epsilon);
        }

        return 0;
    }

    template <typename Real>
    __global__ void detect_body_collision_kernel(
        CollisionInfo<Real> *collisions,
        int *collision_count,
        unsigned int max_collisions,
        unsigned int num_bodies,
        const Real *position,
        const Real *orientation,
        const int *shape,
        const Real *shape_param)
    {
        unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int total_pairs = num_bodies * (num_bodies - 1) / 2;

        if (idx >= total_pairs)
            return;

        int i = (int)((2.0 * num_bodies - 1 - sqrt((2.0 * num_bodies - 1.0) * (2.0 * num_bodies - 1.0) - 8.0 * idx)) / 2.0);
        int j = i + 1 + (idx - i * (num_bodies - 1) + i * (i - 1) / 2);

        const Real *pos_a = &position[i * 3];
        const Real *orient_a = &orientation[i * 4];
        int shape_a = shape[i];
        const Real *params_a = &shape_param[i * 3];

        const Real *pos_b = &position[j * 3];
        const Real *orient_b = &orientation[j * 4];
        int shape_b = shape[j];
        const Real *params_b = &shape_param[j * 3];

        detect_collision(collisions, collision_count, max_collisions, static_cast<int>(i), static_cast<int>(j),
                         shape_a, pos_a, orient_a, params_a,
                         shape_b, pos_b, orient_b, params_b);
    }
}

template <typename Real>
void BodyCollision<Real>::Detect(
    CollisionData<Real> *collision_data,
    unsigned int max_collisions,
    unsigned int num_bodies,
    const Real *dev_position,
    const Real *dev_orientation,
    const int *dev_shape,
    const Real *dev_shape_param)
{
    if (num_bodies < 2 || max_collisions == 0)
        return;
    unsigned int total_pairs = num_bodies * (num_bodies - 1) / 2;

    BodyCollisionKernel::detect_body_collision_kernel<Real><<<CUDA_GRID_SIZE(total_pairs), CUDA_BLOCK_SIZE>>>(
        collision_data->dev_collisions,
        collision_data->dev_collision_count,
        max_collisions,
        num_bodies,
        dev_position,
        dev_orientation,
        dev_shape,
        dev_shape_param);
}

template class BodyCollision<float>;
template class BodyCollision<double>;

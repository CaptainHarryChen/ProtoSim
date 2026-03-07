#pragma once
#include "CollisionData.cuh"

namespace CollisionDataUtils
{
    template <typename Real>
    __device__ int add_collision_info(
        CollisionInfo<Real> *collisions,
        int *collision_count,
        unsigned int max_collisions,
        int body_id_a, int body_id_b,
        const Real *contact_a, const Real *contact_b,
        const Real *normal)
    {
        int slot = atomicAdd(collision_count, 1);
        if (slot >= static_cast<int>(max_collisions))
            return 0;

        collisions[slot].body_id_a = body_id_a;
        collisions[slot].body_id_b = body_id_b;
        collisions[slot].local_point_a[0] = contact_a[0];
        collisions[slot].local_point_a[1] = contact_a[1];
        collisions[slot].local_point_a[2] = contact_a[2];
        collisions[slot].local_point_b[0] = contact_b[0];
        collisions[slot].local_point_b[1] = contact_b[1];
        collisions[slot].local_point_b[2] = contact_b[2];
        collisions[slot].normal[0] = normal[0];
        collisions[slot].normal[1] = normal[1];
        collisions[slot].normal[2] = normal[2];

        return 1;
    }
}

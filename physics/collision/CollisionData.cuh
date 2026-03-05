#pragma once

#include <cuda_runtime.h>

enum RigidBodyShape
{
    RIGID_BODY_BOX = 0,
    RIGID_BODY_SPHERE = 1,
    RIGID_BODY_CAPSULE = 2
};

template <typename Real>
struct CollisionInfo
{
    int body_id_a;
    int body_id_b;
    Real local_point_a[3];
    Real local_point_b[3];
    Real normal[3];
};

template <typename Real>
struct CollisionData
{
    CollisionInfo<Real>* dev_collisions;
    int* dev_collision_count;
    unsigned int max_collisions;
};

extern template struct CollisionInfo<float>;
extern template struct CollisionInfo<double>;
extern template struct CollisionData<float>;
extern template struct CollisionData<double>;

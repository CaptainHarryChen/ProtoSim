#pragma once

#include <collision/CollisionData.cuh>

template <typename Real>
class BodyCollision
{
public:
    BodyCollision() = default;
    ~BodyCollision() = default;

    void Detect(
        CollisionData<Real>* collision_data,
        unsigned int max_collisions,
        unsigned int num_bodies,
        const Real* dev_position,
        const Real* dev_orientation,
        const int* dev_shape,
        const Real* dev_shape_param);
};

extern template class BodyCollision<float>;
extern template class BodyCollision<double>;

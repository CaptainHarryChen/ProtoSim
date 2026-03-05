#pragma once

#include <collision/CollisionData.cuh>

template <typename Real>
class GroundCollision
{
public:
    GroundCollision(Real ground_height = static_cast<Real>(0.0));
    ~GroundCollision();

    void Detect(
        CollisionData<Real>* collision_data,
        unsigned int max_collisions,
        unsigned int num_bodies,
        const Real* dev_position,
        const Real* dev_orientation,
        const int* dev_shape,
        const Real* dev_shape_param
    );

private:
    Real m_ground_height;
};

extern template class GroundCollision<float>;
extern template class GroundCollision<double>;

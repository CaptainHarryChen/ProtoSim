#pragma once

#include <collision/CollisionData.cuh>

template <typename Real>
class GroundCollision
{
public:
    GroundCollision(unsigned int num_bodies, Real ground_height = static_cast<Real>(0.0));
    ~GroundCollision();

    void Detect(
        const Real* dev_position,
        const Real* dev_orientation,
        const int* dev_shape,
        const Real* dev_shape_param
    );

    CollisionData<Real>& GetData() { return m_data; }
    const CollisionData<Real>& GetData() const { return m_data; }

private:
    CollisionData<Real> m_data;
    unsigned int m_num_bodies;
    Real m_ground_height;
};

extern template class GroundCollision<float>;
extern template class GroundCollision<double>;

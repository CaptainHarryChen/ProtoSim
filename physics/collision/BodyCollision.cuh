#pragma once

#include <collision/CollisionData.cuh>

template <typename Real>
class BodyCollision
{
public:
    BodyCollision(unsigned int num_bodies, unsigned int max_collisions);
    ~BodyCollision();

    void Detect(
        const Real* dev_position,
        const Real* dev_orientation,
        const int* dev_shape,
        const Real* dev_shape_param);

    CollisionData<Real>& GetData() { return m_data; }
    const CollisionData<Real>& GetData() const { return m_data; }

private:
    CollisionData<Real> m_data;
    unsigned int m_num_bodies;
};

extern template class BodyCollision<float>;
extern template class BodyCollision<double>;

#pragma once

#include <collision/CollisionData.cuh>

template <typename Real>
class CollisionManager
{
public:
    CollisionManager(unsigned int num_bodies, unsigned int max_collisions_per_type);
    ~CollisionManager();

    void Clear();

    CollisionData<Real>& GetGroundCollisionData() { return m_ground_collisions; }
    CollisionData<Real>& GetBodyCollisionData() { return m_body_collisions; }

    CollisionInfo<Real>* GetAllCollisions() { return m_all_collisions; }
    int* GetAllCollisionCount() { return m_all_collision_count; }
    unsigned int GetMaxCollisions() const { return m_max_total_collisions; }

    void MergeCollisions();

private:
    CollisionData<Real> m_ground_collisions;
    CollisionData<Real> m_body_collisions;

    CollisionInfo<Real>* m_all_collisions;
    int* m_all_collision_count;
    unsigned int m_num_bodies;
    unsigned int m_max_collisions_per_type;
    unsigned int m_max_total_collisions;
};

extern template class CollisionManager<float>;
extern template class CollisionManager<double>;

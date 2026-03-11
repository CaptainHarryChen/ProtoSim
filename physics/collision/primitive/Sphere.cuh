#pragma once
#include <Math/algebra.cuh>
#include <collision/CollisionDataUtils.cuh>

namespace BodyCollisionKernel
{

    template <typename Real>
    __device__ int collide_sphere_sphere(
        CollisionInfo<Real> *collisions,
        int *collision_count,
        unsigned int max_collisions,
        int body_id_a, int body_id_b,
        const Real *pos_a, const Real *orient_a, Real radius_a,
        const Real *pos_b, const Real *orient_b, Real radius_b,
        Real epsilon)
    {
        Real diff[3];
        cudaPhysics::vecSubs3(diff, pos_b, pos_a);
        Real dist = cudaPhysics::len3(diff);

        Real sum_radius = radius_a + radius_b;

        if (dist >= sum_radius || dist < epsilon)
            return 0;

        Real n[3];
        cudaPhysics::vecMul3(n, static_cast<Real>(1.0) / dist, diff);

        Real contact_on_a[3];
        cudaPhysics::vecMul3(contact_on_a, radius_a, n);
        cudaPhysics::vecAdd3(contact_on_a, pos_a, contact_on_a);

        Real contact_on_b[3];
        cudaPhysics::vecMul3(contact_on_b, -radius_b, n);
        cudaPhysics::vecAdd3(contact_on_b, pos_b, contact_on_b);

        Real orient_a_conj[4], orient_b_conj[4];
        cudaPhysics::quatConjugate(orient_a_conj, orient_a);
        cudaPhysics::quatConjugate(orient_b_conj, orient_b);

        Real contact_a[3], contact_b[3];
        cudaPhysics::get_local_point(contact_a, pos_a, orient_a_conj, contact_on_a);
        cudaPhysics::get_local_point(contact_b, pos_b, orient_b_conj, contact_on_b);

        return CollisionDataUtils::add_collision_info(collisions, collision_count, max_collisions, body_id_a, body_id_b, contact_a, contact_b, n);
    }

    template <typename Real>
    __device__ int collide_sphere_box(
        CollisionInfo<Real> *collisions,
        int *collision_count,
        unsigned int max_collisions,
        int body_id_sphere, int body_id_box,
        const Real *pos_sphere, const Real *orient_sphere, Real radius,
        const Real *pos_box, const Real *orient_box, Real hx, Real hy, Real hz,
        Real epsilon)
    {
        Real orient_box_conj[4];
        cudaPhysics::quatConjugate(orient_box_conj, orient_box);

        Real sphere_center_local[3];
        Real diff[3];
        cudaPhysics::vecSubs3(diff, pos_sphere, pos_box);
        cudaPhysics::quatRotateVector(sphere_center_local, orient_box_conj, diff);

        Real closest_local[3];
        closest_local[0] = max(-hx, min(hx, sphere_center_local[0]));
        closest_local[1] = max(-hy, min(hy, sphere_center_local[1]));
        closest_local[2] = max(-hz, min(hz, sphere_center_local[2]));

        Real diff_local[3];
        cudaPhysics::vecSubs3(diff_local, closest_local, sphere_center_local);
        Real dist = cudaPhysics::len3(diff_local);

        if (dist >= radius)
            return 0;

        Real n_local[3];
        if (dist > epsilon)
        {
            cudaPhysics::vecMul3(n_local, static_cast<Real>(1.0) / dist, diff_local);
        }
        else
        {
            Real abs_x = abs(closest_local[0] - hx);
            Real abs_y = abs(closest_local[1] - hy);
            Real abs_z = abs(closest_local[2] - hz);

            if (abs_x <= abs_y && abs_x <= abs_z)
            {
                n_local[0] = sphere_center_local[0] > 0 ? static_cast<Real>(1.0) : static_cast<Real>(-1.0);
                n_local[1] = static_cast<Real>(0.0);
                n_local[2] = static_cast<Real>(0.0);
            }
            else if (abs_y <= abs_z)
            {
                n_local[0] = static_cast<Real>(0.0);
                n_local[1] = sphere_center_local[1] > 0 ? static_cast<Real>(1.0) : static_cast<Real>(-1.0);
                n_local[2] = static_cast<Real>(0.0);
            }
            else
            {
                n_local[0] = static_cast<Real>(0.0);
                n_local[1] = static_cast<Real>(0.0);
                n_local[2] = sphere_center_local[2] > 0 ? static_cast<Real>(1.0) : static_cast<Real>(-1.0);
            }
        }

        Real n_world[3];
        cudaPhysics::quatRotateVector(n_world, orient_box, n_local);

        Real contact_on_sphere[3];
        cudaPhysics::vecMul3(contact_on_sphere, radius, n_world);
        cudaPhysics::vecAdd3(contact_on_sphere, pos_sphere, contact_on_sphere);

        Real contact_on_box[3];
        cudaPhysics::quatRotateVector(contact_on_box, orient_box, closest_local);
        cudaPhysics::vecAdd3(contact_on_box, pos_box, contact_on_box);

        Real orient_sphere_conj[4];
        cudaPhysics::quatConjugate(orient_sphere_conj, orient_sphere);

        Real contact_a[3], contact_b[3];
        cudaPhysics::get_local_point(contact_a, pos_sphere, orient_sphere_conj, contact_on_sphere);
        cudaPhysics::get_local_point(contact_b, pos_box, orient_box_conj, contact_on_box);

        return CollisionDataUtils::add_collision_info(collisions, collision_count, max_collisions, body_id_sphere, body_id_box, contact_a, contact_b, n_world);
    }
}

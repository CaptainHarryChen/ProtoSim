#pragma once
#include <Math/algebra.cuh>
#include <collision/CollisionDataUtils.cuh>

namespace BodyCollisionKernel
{
    template <typename Real>
    __device__ int collide_capsule_sphere(
        CollisionInfo<Real> *collisions,
        int *collision_count,
        unsigned int max_collisions,
        int body_id_capsule, int body_id_sphere,
        const Real *pos_capsule, const Real *orient_capsule, Real radius_capsule, Real half_height,
        const Real *pos_sphere, const Real *orient_sphere, Real radius_sphere,
        Real epsilon,
        bool swap_order)
    {
        Real axis[3] = {static_cast<Real>(0.0), static_cast<Real>(1.0), static_cast<Real>(0.0)};
        Real rotated_axis[3];
        cudaPhysics::quatRotateVector(rotated_axis, orient_capsule, axis);

        Real top_center[3], bottom_center[3];
        cudaPhysics::axpby(top_center, static_cast<Real>(1.0), pos_capsule, half_height, rotated_axis, 3);
        cudaPhysics::axpby(bottom_center, static_cast<Real>(1.0), pos_capsule, -half_height, rotated_axis, 3);

        Real closest_on_segment[3];
        {
            Real ab[3];
            cudaPhysics::vecSubs3(ab, top_center, bottom_center);
            Real ab_len_sq = cudaPhysics::dot3(ab, ab);

            if (ab_len_sq < epsilon)
            {
                cudaPhysics::vecCopy3(closest_on_segment, bottom_center);
            }
            else
            {
                Real ap[3];
                cudaPhysics::vecSubs3(ap, pos_sphere, bottom_center);
                Real t = cudaPhysics::dot3(ap, ab) / ab_len_sq;
                t = max(static_cast<Real>(0.0), min(static_cast<Real>(1.0), t));
                cudaPhysics::axpby(closest_on_segment, static_cast<Real>(1.0), bottom_center, t, ab, 3);
            }
        }

        Real diff[3];
        cudaPhysics::vecSubs3(diff, pos_sphere, closest_on_segment);
        Real dist = cudaPhysics::len3(diff);

        Real sum_radius = radius_capsule + radius_sphere;

        if (dist >= sum_radius || dist < epsilon)
            return 0;

        Real n[3];
        cudaPhysics::vecMul3(n, static_cast<Real>(1.0) / dist, diff);

        Real contact_on_capsule[3];
        cudaPhysics::vecMul3(contact_on_capsule, radius_capsule, n);
        cudaPhysics::vecAdd3(contact_on_capsule, closest_on_segment, contact_on_capsule);

        Real contact_on_sphere[3];
        cudaPhysics::vecMul3(contact_on_sphere, -radius_sphere, n);
        cudaPhysics::vecAdd3(contact_on_sphere, pos_sphere, contact_on_sphere);

        Real orient_capsule_conj[4], orient_sphere_conj[4];
        cudaPhysics::quatConjugate(orient_capsule_conj, orient_capsule);
        cudaPhysics::quatConjugate(orient_sphere_conj, orient_sphere);

        Real contact_a[3], contact_b[3];
        cudaPhysics::get_local_point(contact_a, pos_capsule, orient_capsule_conj, contact_on_capsule);
        cudaPhysics::get_local_point(contact_b, pos_sphere, orient_sphere_conj, contact_on_sphere);

        if (swap_order)
        {
            Real neg_n[3];
            cudaPhysics::vecMul3(neg_n, static_cast<Real>(-1.0), n);
            return CollisionDataUtils::add_collision_info(collisions, collision_count, max_collisions, body_id_sphere, body_id_capsule, contact_b, contact_a, neg_n);
        }
        else
        {
            return CollisionDataUtils::add_collision_info(collisions, collision_count, max_collisions, body_id_capsule, body_id_sphere, contact_a, contact_b, n);
        }
    }

    template <typename Real>
    __device__ int collide_capsule_box(
        CollisionInfo<Real> *collisions,
        int *collision_count,
        unsigned int max_collisions,
        int body_id_capsule, int body_id_box,
        const Real *pos_capsule, const Real *orient_capsule, Real radius_capsule, Real half_height,
        const Real *pos_box, const Real *orient_box, Real hx, Real hy, Real hz,
        Real epsilon,
        bool swap_order)
    {
        Real axis[3] = {static_cast<Real>(0.0), static_cast<Real>(1.0), static_cast<Real>(0.0)};
        Real rotated_axis[3];
        cudaPhysics::quatRotateVector(rotated_axis, orient_capsule, axis);

        Real top_center[3], bottom_center[3];
        cudaPhysics::axpby(top_center, static_cast<Real>(1.0), pos_capsule, half_height, rotated_axis, 3);
        cudaPhysics::axpby(bottom_center, static_cast<Real>(1.0), pos_capsule, -half_height, rotated_axis, 3);

        Real orient_box_conj[4];
        cudaPhysics::quatConjugate(orient_box_conj, orient_box);

        Real top_local[3], bottom_local[3];
        Real diff_top[3], diff_bottom[3];
        cudaPhysics::vecSubs3(diff_top, top_center, pos_box);
        cudaPhysics::vecSubs3(diff_bottom, bottom_center, pos_box);
        cudaPhysics::quatRotateVector(top_local, orient_box_conj, diff_top);
        cudaPhysics::quatRotateVector(bottom_local, orient_box_conj, diff_bottom);

        Real ab_local[3];
        cudaPhysics::vecSubs3(ab_local, top_local, bottom_local);
        Real ab_len_sq = cudaPhysics::dot3(ab_local, ab_local);

        Real best_dist = static_cast<Real>(1e30);
        Real best_point_local[3];
        Real best_t = static_cast<Real>(0.5);
        Real best_segment_local[3];

        int num_samples = 32;
        for (int i = 0; i <= num_samples; ++i)
        {
            Real t = static_cast<Real>(i) / static_cast<Real>(num_samples);
            Real point_local[3];
            cudaPhysics::axpby(point_local, static_cast<Real>(1.0) - t, bottom_local, t, top_local, 3);

            Real clamped[3];
            clamped[0] = max(-hx, min(hx, point_local[0]));
            clamped[1] = max(-hy, min(hy, point_local[1]));
            clamped[2] = max(-hz, min(hz, point_local[2]));

            Real diff[3];
            cudaPhysics::vecSubs3(diff, point_local, clamped);
            Real dist = cudaPhysics::len3(diff);

            if (dist < best_dist)
            {
                best_dist = dist;
                cudaPhysics::vecCopy3(best_point_local, clamped);
                cudaPhysics::vecCopy3(best_segment_local, point_local);
                best_t = t;
            }
        }

        if (best_dist >= radius_capsule)
            return 0;

        Real n_local[3];
        if (best_dist > epsilon)
        {
            Real diff[3];
            cudaPhysics::vecSubs3(diff, best_segment_local, best_point_local);
            Real inv_dist = static_cast<Real>(1.0) / best_dist;
            cudaPhysics::vecMul3(n_local, inv_dist, diff);
        }
        else
        {
            Real abs_x = abs(best_point_local[0] - hx);
            Real abs_y = abs(best_point_local[1] - hy);
            Real abs_z = abs(best_point_local[2] - hz);

            if (abs_x <= abs_y && abs_x <= abs_z)
            {
                n_local[0] = best_segment_local[0] > 0 ? static_cast<Real>(1.0) : static_cast<Real>(-1.0);
                n_local[1] = static_cast<Real>(0.0);
                n_local[2] = static_cast<Real>(0.0);
            }
            else if (abs_y <= abs_z)
            {
                n_local[0] = static_cast<Real>(0.0);
                n_local[1] = best_segment_local[1] > 0 ? static_cast<Real>(1.0) : static_cast<Real>(-1.0);
                n_local[2] = static_cast<Real>(0.0);
            }
            else
            {
                n_local[0] = static_cast<Real>(0.0);
                n_local[1] = static_cast<Real>(0.0);
                n_local[2] = best_segment_local[2] > 0 ? static_cast<Real>(1.0) : static_cast<Real>(-1.0);
            }
        }

        Real n_world[3];
        cudaPhysics::quatRotateVector(n_world, orient_box, n_local);

        Real contact_on_box[3];
        cudaPhysics::quatRotateVector(contact_on_box, orient_box, best_point_local);
        cudaPhysics::vecAdd3(contact_on_box, pos_box, contact_on_box);

        Real contact_on_capsule[3];
        cudaPhysics::vecMul3(contact_on_capsule, -radius_capsule, n_world);
        cudaPhysics::vecAdd3(contact_on_capsule, contact_on_box, contact_on_capsule);

        Real orient_capsule_conj[4];
        cudaPhysics::quatConjugate(orient_capsule_conj, orient_capsule);

        Real contact_a[3], contact_b[3];
        cudaPhysics::get_local_point(contact_a, pos_capsule, orient_capsule_conj, contact_on_capsule);
        cudaPhysics::get_local_point(contact_b, pos_box, orient_box_conj, contact_on_box);

        Real normal_out[3];
        cudaPhysics::vecMul3(normal_out, static_cast<Real>(-1.0), n_world);

        int num_contacts = 0;

        if (swap_order)
        {
            num_contacts += CollisionDataUtils::add_collision_info(collisions, collision_count, max_collisions, body_id_box, body_id_capsule, contact_b, contact_a, n_world);
        }
        else
        {
            num_contacts += CollisionDataUtils::add_collision_info(collisions, collision_count, max_collisions, body_id_capsule, body_id_box, contact_a, contact_b, normal_out);
        }

        Real t_threshold = static_cast<Real>(0.1);
        if (best_t > t_threshold && best_t < static_cast<Real>(1.0) - t_threshold)
        {
            Real tangent[3];
            cudaPhysics::cross3(tangent, n_world, rotated_axis);
            Real tangent_len = cudaPhysics::len3(tangent);

            if (tangent_len > epsilon)
            {
                cudaPhysics::vecMul3(tangent, static_cast<Real>(1.0) / tangent_len, tangent);

                Real offset = min(static_cast<Real>(0.5) * half_height, radius_capsule * static_cast<Real>(0.5));

                Real contact2_on_capsule[3];
                contact2_on_capsule[0] = contact_on_capsule[0] + offset * tangent[0];
                contact2_on_capsule[1] = contact_on_capsule[1] + offset * tangent[1];
                contact2_on_capsule[2] = contact_on_capsule[2] + offset * tangent[2];

                Real contact2_local_capsule[3];
                cudaPhysics::get_local_point(contact2_local_capsule, pos_capsule, orient_capsule_conj, contact2_on_capsule);

                Real contact2_on_box[3];
                contact2_on_box[0] = contact_on_box[0] + offset * tangent[0];
                contact2_on_box[1] = contact_on_box[1] + offset * tangent[1];
                contact2_on_box[2] = contact_on_box[2] + offset * tangent[2];

                Real contact2_local_box[3];
                cudaPhysics::get_local_point(contact2_local_box, pos_box, orient_box_conj, contact2_on_box);

                if (swap_order)
                {
                    num_contacts += CollisionDataUtils::add_collision_info(collisions, collision_count, max_collisions, body_id_box, body_id_capsule, contact2_local_box, contact2_local_capsule, n_world);
                }
                else
                {
                    num_contacts += CollisionDataUtils::add_collision_info(collisions, collision_count, max_collisions, body_id_capsule, body_id_box, contact2_local_capsule, contact2_local_box, normal_out);
                }
            }
        }

        return num_contacts;
    }

    template <typename Real>
    __device__ int collide_capsule_capsule(
        CollisionInfo<Real> *collisions,
        int *collision_count,
        unsigned int max_collisions,
        int body_id_a, int body_id_b,
        const Real *pos_a, const Real *orient_a, Real radius_a, Real half_height_a,
        const Real *pos_b, const Real *orient_b, Real radius_b, Real half_height_b,
        Real epsilon)
    {
        Real axis_a[3] = {static_cast<Real>(0.0), static_cast<Real>(1.0), static_cast<Real>(0.0)};
        Real rotated_axis_a[3];
        cudaPhysics::quatRotateVector(rotated_axis_a, orient_a, axis_a);

        Real top_a[3], bottom_a[3];
        cudaPhysics::axpby(top_a, static_cast<Real>(1.0), pos_a, half_height_a, rotated_axis_a, 3);
        cudaPhysics::axpby(bottom_a, static_cast<Real>(1.0), pos_a, -half_height_a, rotated_axis_a, 3);

        Real axis_b[3] = {static_cast<Real>(0.0), static_cast<Real>(1.0), static_cast<Real>(0.0)};
        Real rotated_axis_b[3];
        cudaPhysics::quatRotateVector(rotated_axis_b, orient_b, axis_b);

        Real top_b[3], bottom_b[3];
        cudaPhysics::axpby(top_b, static_cast<Real>(1.0), pos_b, half_height_b, rotated_axis_b, 3);
        cudaPhysics::axpby(bottom_b, static_cast<Real>(1.0), pos_b, -half_height_b, rotated_axis_b, 3);

        Real d1[3], d2[3], r[3];
        cudaPhysics::vecSubs3(d1, top_a, bottom_a);
        cudaPhysics::vecSubs3(d2, top_b, bottom_b);
        cudaPhysics::vecSubs3(r, bottom_a, bottom_b);

        Real a = cudaPhysics::dot3(d1, d1);
        Real b = cudaPhysics::dot3(d1, d2);
        Real c = cudaPhysics::dot3(d2, d2);
        Real d = cudaPhysics::dot3(d1, r);
        Real e = cudaPhysics::dot3(d2, r);

        Real denom = a * c - b * b;

        Real s, t;
        if (denom < epsilon)
        {
            s = static_cast<Real>(0.0);
            t = (b > c ? d / b : e / c);
        }
        else
        {
            s = (b * e - c * d) / denom;
            t = (a * e - b * d) / denom;
        }

        s = max(static_cast<Real>(0.0), min(static_cast<Real>(1.0), s));
        t = max(static_cast<Real>(0.0), min(static_cast<Real>(1.0), t));

        Real closest_a[3], closest_b[3];
        cudaPhysics::axpby(closest_a, static_cast<Real>(1.0), bottom_a, s, d1, 3);
        cudaPhysics::axpby(closest_b, static_cast<Real>(1.0), bottom_b, t, d2, 3);

        Real diff[3];
        cudaPhysics::vecSubs3(diff, closest_b, closest_a);
        Real dist = cudaPhysics::len3(diff);

        Real sum_radius = radius_a + radius_b;

        if (dist >= sum_radius || dist < epsilon)
            return 0;

        Real n[3];
        cudaPhysics::vecMul3(n, static_cast<Real>(1.0) / dist, diff);

        Real contact_on_a[3];
        cudaPhysics::vecMul3(contact_on_a, radius_a, n);
        cudaPhysics::vecAdd3(contact_on_a, closest_a, contact_on_a);

        Real contact_on_b[3];
        cudaPhysics::vecMul3(contact_on_b, -radius_b, n);
        cudaPhysics::vecAdd3(contact_on_b, closest_b, contact_on_b);

        Real orient_a_conj[4], orient_b_conj[4];
        cudaPhysics::quatConjugate(orient_a_conj, orient_a);
        cudaPhysics::quatConjugate(orient_b_conj, orient_b);

        Real contact_a[3], contact_b[3];
        cudaPhysics::get_local_point(contact_a, pos_a, orient_a_conj, contact_on_a);
        cudaPhysics::get_local_point(contact_b, pos_b, orient_b_conj, contact_on_b);

        return CollisionDataUtils::add_collision_info(collisions, collision_count, max_collisions, body_id_a, body_id_b, contact_a, contact_b, n);
    }
}

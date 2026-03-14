#pragma once
#include <Math/algebra.cuh>
#include <Math/geometry.cuh>
#include <collision/CollisionDataUtils.cuh>
#include <collision/primitive/Box.cuh>

namespace BodyCollisionKernel
{
    template <typename Real>
    __device__ void project_capsule_on_axis(Real &min_proj, Real &max_proj,
                                             const Real *capsule_center, const Real *capsule_axis, Real half_height, Real radius,
                                             const Real *axis)
    {
        Real cap_proj = cudaPhysics::dot3(capsule_center, axis);
        Real axis_proj = cudaPhysics::dot3(capsule_axis, axis);
        Real half_h_proj = fabs(axis_proj) * half_height;

        min_proj = cap_proj - half_h_proj - radius;
        max_proj = cap_proj + half_h_proj + radius;
    }

    template <typename Real>
    __device__ bool test_axis_capsule_box(Real &overlap,
                                           const Real *axis,
                                           const Real *capsule_center, const Real *capsule_axis, Real half_height, Real radius,
                                           const Real *box_pos, const Real *box_orient, Real hx, Real hy, Real hz,
                                           Real epsilon)
    {
        Real len = cudaPhysics::len3(axis);
        if (len < epsilon)
            return true;

        Real n[3];
        cudaPhysics::vecMul3(n, static_cast<Real>(1.0) / len, axis);

        Real min_cap, max_cap;
        project_capsule_on_axis(min_cap, max_cap, capsule_center, capsule_axis, half_height, radius, n);

        Real min_box, max_box;
        project_box_on_axis(min_box, max_box, box_pos, box_orient, hx, hy, hz, n);

        if (max_cap < min_box || max_box < min_cap)
            return false;

        Real o1 = max_cap - min_box;
        Real o2 = max_box - min_cap;
        overlap = min(o1, o2);

        return true;
    }

    template <typename Real>
    __device__ int generate_capsule_face_contacts(
        CollisionInfo<Real> *collisions,
        int *collision_count,
        unsigned int max_collisions,
        int body_id_capsule, int body_id_box,
        const Real *capsule_pos, const Real *capsule_axis, Real half_height, Real radius,
        const Real *capsule_orient_conj,
        const Real *box_pos, const Real *box_orient, const Real *box_extents,
        const Real *box_orient_conj,
        const Real box_axes[3][3],
        int ref_face_idx,
        const Real *n,
        Real epsilon)
    {
        int ref_axis = ref_face_idx % 3;
        int ref_sign = (ref_face_idx < 3) ? 1 : -1;

        Real ref_normal[3];
        cudaPhysics::vecCopy3(ref_normal, box_axes[ref_axis]);
        if (ref_sign < 0)
            cudaPhysics::vecMul3(ref_normal, static_cast<Real>(-1.0), ref_normal);

        int t1 = (ref_axis + 1) % 3;
        int t2 = (ref_axis + 2) % 3;

        Real ref_face_center[3];
        cudaPhysics::axpby(ref_face_center, static_cast<Real>(1.0), box_pos, box_extents[ref_axis], ref_normal, 3);

        Real capsule_center[3];
        cudaPhysics::vecCopy3(capsule_center, capsule_pos);

        Real capsule_top[3], capsule_bottom[3];
        cudaPhysics::axpby(capsule_top, static_cast<Real>(1.0), capsule_pos, half_height, capsule_axis, 3);
        cudaPhysics::axpby(capsule_bottom, static_cast<Real>(1.0), capsule_pos, -half_height, capsule_axis, 3);

        Real side_normals[4][3];
        Real side_points[4][3];

        cudaPhysics::vecCopy3(side_normals[0], box_axes[t1]);
        cudaPhysics::vecMul3(side_normals[1], static_cast<Real>(-1.0), box_axes[t1]);
        cudaPhysics::vecCopy3(side_normals[2], box_axes[t2]);
        cudaPhysics::vecMul3(side_normals[3], static_cast<Real>(-1.0), box_axes[t2]);

        cudaPhysics::axpby(side_points[0], static_cast<Real>(1.0), ref_face_center, box_extents[t1], box_axes[t1], 3);
        cudaPhysics::axpby(side_points[1], static_cast<Real>(1.0), ref_face_center, -box_extents[t1], box_axes[t1], 3);
        cudaPhysics::axpby(side_points[2], static_cast<Real>(1.0), ref_face_center, box_extents[t2], box_axes[t2], 3);
        cudaPhysics::axpby(side_points[3], static_cast<Real>(1.0), ref_face_center, -box_extents[t2], box_axes[t2], 3);

        Real segment[2][3];
        cudaPhysics::vecCopy3(segment[0], capsule_top);
        cudaPhysics::vecCopy3(segment[1], capsule_bottom);

        Real clipped[2][3];
        int num_points = 2;

        for (int i = 0; i < 4 && num_points > 0; ++i)
        {
            int output_count = 0;
            for (int j = 0; j < num_points; ++j)
            {
                int k = (j + 1) % num_points;
                Real d_j = cudaPhysics::dot3(segment[j], side_normals[i]) - cudaPhysics::dot3(side_points[i], side_normals[i]);
                Real d_k = cudaPhysics::dot3(segment[k], side_normals[i]) - cudaPhysics::dot3(side_points[i], side_normals[i]);

                bool inside_j = (d_j <= epsilon);
                bool inside_k = (d_k <= epsilon);

                if (inside_j)
                {
                    cudaPhysics::vecCopy3(clipped[output_count], segment[j]);
                    output_count++;
                }

                if (inside_j != inside_k)
                {
                    Real t = d_j / (d_j - d_k);
                    cudaPhysics::axpby(clipped[output_count], static_cast<Real>(1.0 - t), segment[j], t, segment[k], 3);
                    output_count++;
                }
            }
            num_points = output_count;
            for (int j = 0; j < num_points; ++j)
                cudaPhysics::vecCopy3(segment[j], clipped[j]);
        }

        int num_contacts = 0;
        for (int i = 0; i < num_points && num_contacts < MAX_MANIFOLD_POINTS; ++i)
        {
            Real face_to_seg[3];
            cudaPhysics::vecSubs3(face_to_seg, segment[i], ref_face_center);
            Real dist_to_plane = cudaPhysics::dot3(ref_normal, face_to_seg);

            if (dist_to_plane < radius)
            {
                Real contact_on_box[3];
                cudaPhysics::axpby(contact_on_box, static_cast<Real>(1.0), segment[i], -dist_to_plane, ref_normal, 3);

                Real contact_on_capsule[3];
                cudaPhysics::axpby(contact_on_capsule, static_cast<Real>(1.0), segment[i], radius, n, 3);

                Real local_capsule[3], local_box[3];
                cudaPhysics::get_local_point(local_capsule, capsule_pos, capsule_orient_conj, contact_on_capsule);
                cudaPhysics::get_local_point(local_box, box_pos, box_orient_conj, contact_on_box);

                num_contacts += CollisionDataUtils::add_collision_info(
                    collisions, collision_count, max_collisions,
                    body_id_capsule, body_id_box,
                    local_capsule, local_box,
                    n);
            }
        }

        return num_contacts;
    }

    template <typename Real>
    __device__ int generate_capsule_edge_contacts(
        CollisionInfo<Real> *collisions,
        int *collision_count,
        unsigned int max_collisions,
        int body_id_capsule, int body_id_box,
        const Real *capsule_pos, const Real *capsule_axis, Real half_height, Real radius,
        const Real *capsule_orient_conj,
        const Real *box_pos, const Real *box_orient, const Real *box_extents,
        const Real *box_orient_conj,
        const Real box_axes[3][3],
        int edge_axis_idx,
        const Real *n,
        Real epsilon)
    {
        int t1 = (edge_axis_idx + 1) % 3;
        int t2 = (edge_axis_idx + 2) % 3;

        Real s1 = cudaPhysics::dot3(n, box_axes[t1]) < 0 ? static_cast<Real>(1.0) : static_cast<Real>(-1.0);
        Real s2 = cudaPhysics::dot3(n, box_axes[t2]) < 0 ? static_cast<Real>(1.0) : static_cast<Real>(-1.0);

        Real local_edge[3] = {static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)};
        local_edge[t1] = s1 * box_extents[t1];
        local_edge[t2] = s2 * box_extents[t2];

        Real box_edge_point[3];
        cudaPhysics::quatRotateVector(box_edge_point, box_orient, local_edge);
        cudaPhysics::vecAdd3(box_edge_point, box_pos, box_edge_point);

        Real dB[3];
        cudaPhysics::vecCopy3(dB, box_axes[edge_axis_idx]);

        Real r[3];
        cudaPhysics::vecSubs3(r, capsule_pos, box_edge_point);

        Real a = cudaPhysics::dot3(capsule_axis, capsule_axis);
        Real e = cudaPhysics::dot3(dB, dB);
        Real b = cudaPhysics::dot3(capsule_axis, dB);
        Real c = cudaPhysics::dot3(capsule_axis, r);
        Real f = cudaPhysics::dot3(dB, r);

        Real denom = a * e - b * b;

        if (fabs(denom) < epsilon)
            return 0;

        Real t = (b * f - c * e) / denom;
        Real s = (a * f - b * c) / denom;

        t = max(-half_height, min(half_height, t));
        s = max(-box_extents[edge_axis_idx], min(box_extents[edge_axis_idx], s));

        Real closest_capsule[3], closest_box[3];
        cudaPhysics::axpby(closest_capsule, static_cast<Real>(1.0), capsule_pos, t, capsule_axis, 3);
        cudaPhysics::axpby(closest_box, static_cast<Real>(1.0), box_edge_point, s, dB, 3);

        Real contact_on_capsule[3];
        cudaPhysics::axpby(contact_on_capsule, static_cast<Real>(1.0), closest_capsule, radius, n, 3);

        Real local_capsule[3], local_box[3];
        cudaPhysics::get_local_point(local_capsule, capsule_pos, capsule_orient_conj, contact_on_capsule);
        cudaPhysics::get_local_point(local_box, box_pos, box_orient_conj, closest_box);

        return CollisionDataUtils::add_collision_info(
            collisions, collision_count, max_collisions,
            body_id_capsule, body_id_box,
            local_capsule, local_box,
            n);
    }

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
        Real epsilon)
    {
        Real capsule_axis[3] = {static_cast<Real>(0.0), static_cast<Real>(1.0), static_cast<Real>(0.0)};
        Real rotated_axis[3];
        cudaPhysics::quatRotateVector(rotated_axis, orient_capsule, capsule_axis);

        Real box_axes[3][3];
        for (int i = 0; i < 3; ++i)
        {
            Real basis[3] = {static_cast<Real>(i == 0), static_cast<Real>(i == 1), static_cast<Real>(i == 2)};
            cudaPhysics::quatRotateVector(box_axes[i], orient_box, basis);
        }

        Real min_overlap = static_cast<Real>(1e30);
        Real min_axis[3];
        int min_axis_type = 0;
        int min_axis_idx = 0;

        for (int i = 0; i < 3; ++i)
        {
            Real overlap;
            if (!test_axis_capsule_box(overlap, box_axes[i],
                                        pos_capsule, rotated_axis, half_height, radius_capsule,
                                        pos_box, orient_box, hx, hy, hz, epsilon))
                return 0;

            if (overlap < min_overlap)
            {
                min_overlap = overlap;
                cudaPhysics::vecCopy3(min_axis, box_axes[i]);
                min_axis_type = 0;
                min_axis_idx = i;
            }
        }

        {
            Real overlap;
            if (!test_axis_capsule_box(overlap, rotated_axis,
                                        pos_capsule, rotated_axis, half_height, radius_capsule,
                                        pos_box, orient_box, hx, hy, hz, epsilon))
                return 0;

            if (overlap < min_overlap)
            {
                min_overlap = overlap;
                cudaPhysics::vecCopy3(min_axis, rotated_axis);
                min_axis_type = 1;
                min_axis_idx = 0;
            }
        }

        for (int i = 0; i < 3; ++i)
        {
            Real cross_axis[3];
            cudaPhysics::cross3(cross_axis, box_axes[i], rotated_axis);

            Real cross_len = cudaPhysics::len3(cross_axis);
            if (cross_len < epsilon)
                continue;

            Real overlap;
            if (!test_axis_capsule_box(overlap, cross_axis,
                                        pos_capsule, rotated_axis, half_height, radius_capsule,
                                        pos_box, orient_box, hx, hy, hz, epsilon))
                return 0;

            if (overlap < min_overlap)
            {
                min_overlap = overlap;
                cudaPhysics::vecCopy3(min_axis, cross_axis);
                min_axis_type = 2;
                min_axis_idx = i;
            }
        }

        if (min_overlap <= epsilon)
            return 0;

        Real d[3];
        cudaPhysics::vecSubs3(d, pos_box, pos_capsule);
        Real sign = cudaPhysics::dot3(d, min_axis) < static_cast<Real>(0.0) ? static_cast<Real>(-1.0) : static_cast<Real>(1.0);
        cudaPhysics::vecMul3(min_axis, sign, min_axis);

        Real len = cudaPhysics::len3(min_axis);
        if (len < epsilon)
            return 0;

        Real n[3];
        cudaPhysics::vecMul3(n, static_cast<Real>(1.0) / len, min_axis);

        Real box_orient_conj[4];
        cudaPhysics::quatConjugate(box_orient_conj, orient_box);

        Real capsule_orient_conj[4];
        cudaPhysics::quatConjugate(capsule_orient_conj, orient_capsule);

        Real box_extents[3] = {hx, hy, hz};

        int num_contacts = 0;

        if (min_axis_type == 0)
        {
            int ref_face_idx = min_axis_idx + (sign < static_cast<Real>(0.0) ? 0 : 3);

            num_contacts = generate_capsule_face_contacts(
                collisions, collision_count, max_collisions,
                body_id_capsule, body_id_box,
                pos_capsule, rotated_axis, half_height, radius_capsule,
                capsule_orient_conj,
                pos_box, orient_box, box_extents,
                box_orient_conj,
                box_axes,
                ref_face_idx,
                n, epsilon);
        }
        else if (min_axis_type == 1)
        {
            Real top[3], bottom[3];
            cudaPhysics::axpby(top, static_cast<Real>(1.0), pos_capsule, half_height, rotated_axis, 3);
            cudaPhysics::axpby(bottom, static_cast<Real>(1.0), pos_capsule, -half_height, rotated_axis, 3);

            Real dist_top_sq = static_cast<Real>(0.0);
            Real dist_bottom_sq = static_cast<Real>(0.0);
            for (int i = 0; i < 3; ++i)
            {
                dist_top_sq += (top[i] - pos_box[i]) * (top[i] - pos_box[i]);
                dist_bottom_sq += (bottom[i] - pos_box[i]) * (bottom[i] - pos_box[i]);
            }

            Real sphere_center[3];
            if (dist_top_sq < dist_bottom_sq)
                cudaPhysics::vecCopy3(sphere_center, top);
            else
                cudaPhysics::vecCopy3(sphere_center, bottom);

            Real local_sphere[3];
            Real diff[3];
            cudaPhysics::vecSubs3(diff, sphere_center, pos_box);
            cudaPhysics::quatRotateVector(local_sphere, box_orient_conj, diff);

            Real clamped[3];
            clamped[0] = max(-hx, min(hx, local_sphere[0]));
            clamped[1] = max(-hy, min(hy, local_sphere[1]));
            clamped[2] = max(-hz, min(hz, local_sphere[2]));

            Real diff_local[3];
            cudaPhysics::vecSubs3(diff_local, local_sphere, clamped);
            Real dist = cudaPhysics::len3(diff_local);

            if (dist >= radius_capsule)
                return 0;

            Real contact_on_box_local[3];
            cudaPhysics::vecCopy3(contact_on_box_local, clamped);

            Real contact_on_box[3];
            cudaPhysics::quatRotateVector(contact_on_box, orient_box, contact_on_box_local);
            cudaPhysics::vecAdd3(contact_on_box, pos_box, contact_on_box);

            Real contact_on_capsule[3];
            cudaPhysics::axpby(contact_on_capsule, static_cast<Real>(1.0), sphere_center, -radius_capsule, n, 3);

            Real local_capsule[3], local_box[3];
            cudaPhysics::get_local_point(local_capsule, pos_capsule, capsule_orient_conj, contact_on_capsule);
            cudaPhysics::get_local_point(local_box, pos_box, box_orient_conj, contact_on_box);

            num_contacts = CollisionDataUtils::add_collision_info(
                collisions, collision_count, max_collisions,
                body_id_capsule, body_id_box,
                local_capsule, local_box,
                n);
        }
        else
        {
            num_contacts = generate_capsule_edge_contacts(
                collisions, collision_count, max_collisions,
                body_id_capsule, body_id_box,
                pos_capsule, rotated_axis, half_height, radius_capsule,
                capsule_orient_conj,
                pos_box, orient_box, box_extents,
                box_orient_conj,
                box_axes,
                min_axis_idx,
                n, epsilon);
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
        Real orient_a_conj[4], orient_b_conj[4];
        cudaPhysics::quatConjugate(orient_a_conj, orient_a);
        cudaPhysics::quatConjugate(orient_b_conj, orient_b);

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

        Real cross_a_b[3];
        cudaPhysics::cross3(cross_a_b, rotated_axis_a, rotated_axis_b);
        Real cross_len = cudaPhysics::len3(cross_a_b);

        Real sum_radius = radius_a + radius_b;
        int num_contacts = 0;

        auto add_contact = [&](const Real *point_on_a, const Real *point_on_b) -> int {
            Real diff[3];
            cudaPhysics::vecSubs3(diff, point_on_b, point_on_a);
            Real dist = cudaPhysics::len3(diff);

            if (dist >= sum_radius || dist < epsilon)
                return 0;

            Real n[3];
            cudaPhysics::vecMul3(n, static_cast<Real>(1.0) / dist, diff);

            Real contact_on_a[3];
            cudaPhysics::vecMul3(contact_on_a, radius_a, n);
            cudaPhysics::vecAdd3(contact_on_a, point_on_a, contact_on_a);

            Real contact_on_b[3];
            cudaPhysics::vecMul3(contact_on_b, -radius_b, n);
            cudaPhysics::vecAdd3(contact_on_b, point_on_b, contact_on_b);

            Real local_a[3], local_b[3];
            cudaPhysics::get_local_point(local_a, pos_a, orient_a_conj, contact_on_a);
            cudaPhysics::get_local_point(local_b, pos_b, orient_b_conj, contact_on_b);

            return CollisionDataUtils::add_collision_info(collisions, collision_count, max_collisions, body_id_a, body_id_b, local_a, local_b, n);
        };

        if (cross_len < epsilon)
        {
            Real closest[3];
            Real dist = cudaPhysics::point_to_segment_distance(closest, top_a, bottom_b, top_b);
            num_contacts += add_contact(top_a, closest);

            dist = cudaPhysics::point_to_segment_distance(closest, bottom_a, bottom_b, top_b);
            num_contacts += add_contact(bottom_a, closest);

            dist = cudaPhysics::point_to_segment_distance(closest, top_b, bottom_a, top_a);
            num_contacts += add_contact(closest, top_b);

            dist = cudaPhysics::point_to_segment_distance(closest, bottom_b, bottom_a, top_a);
            num_contacts += add_contact(closest, bottom_b);
        }
        else
        {
            Real closest_a[3], closest_b[3];
            Real dist = cudaPhysics::segment_to_segment_distance(closest_a, closest_b, bottom_a, top_a, bottom_b, top_b);
            num_contacts += add_contact(closest_a, closest_b);
        }

        return num_contacts;
    }
}

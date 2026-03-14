#pragma once
#include <Math/algebra.cuh>
#include <collision/CollisionDataUtils.cuh>

namespace BodyCollisionKernel
{
    template <typename Real>
    __device__ void get_box_axis(Real *axis, const Real *orient, int axis_idx)
    {
        Real basis[3] = {static_cast<Real>(axis_idx == 0),
                         static_cast<Real>(axis_idx == 1),
                         static_cast<Real>(axis_idx == 2)};
        cudaPhysics::quatRotateVector(axis, orient, basis);
    }

    template <typename Real>
    __device__ void project_box_on_axis(Real &min_proj, Real &max_proj, const Real *pos, const Real *orient, Real hx, Real hy, Real hz, const Real *axis)
    {
        Real corners[8][3] = {
            {-hx, -hy, -hz}, {hx, -hy, -hz}, {-hx, hy, -hz}, {hx, hy, -hz}, {-hx, -hy, hz}, {hx, -hy, hz}, {-hx, hy, hz}, {hx, hy, hz}};

        min_proj = static_cast<Real>(1e30);
        max_proj = static_cast<Real>(-1e30);

        for (int i = 0; i < 8; ++i)
        {
            Real world_corner[3];
            cudaPhysics::quatRotateVector(world_corner, orient, corners[i]);
            Real world_point[3];
            cudaPhysics::vecAdd3(world_point, pos, world_corner);

            Real proj = cudaPhysics::dot3(world_point, axis);
            min_proj = min(min_proj, proj);
            max_proj = max(max_proj, proj);
        }
    }

    template <typename Real>
    __device__ bool test_axis(Real &overlap, const Real *axis,
                              const Real *pos_a, const Real *orient_a, Real hx_a, Real hy_a, Real hz_a,
                              const Real *pos_b, const Real *orient_b, Real hx_b, Real hy_b, Real hz_b,
                              Real epsilon)
    {
        Real len = cudaPhysics::len3(axis);
        if (len < epsilon)
            return true;

        Real n[3];
        cudaPhysics::vecMul3(n, static_cast<Real>(1.0) / len, axis);

        Real min_a, max_a, min_b, max_b;
        project_box_on_axis(min_a, max_a, pos_a, orient_a, hx_a, hy_a, hz_a, n);
        project_box_on_axis(min_b, max_b, pos_b, orient_b, hx_b, hy_b, hz_b, n);

        if (max_a < min_b || max_b < min_a)
            return false;

        Real o1 = max_a - min_b;
        Real o2 = max_b - min_a;
        overlap = min(o1, o2);

        return true;
    }

    template <typename Real>
    __device__ void get_face_vertices(Real vertices[4][3], const Real *pos, const Real *orient, const Real *extents, int face_idx)
    {
        int axis = face_idx % 3;
        int sign = (face_idx < 3) ? 1 : -1;

        int t1 = (axis + 1) % 3;
        int t2 = (axis + 2) % 3;

        Real face_ext = extents[axis] * sign;

        for (int i = 0; i < 4; ++i)
        {
            int s1 = (i == 1 || i == 2) ? 1 : -1;
            int s2 = (i == 2 || i == 3) ? 1 : -1;

            Real local_vert[3];
            local_vert[axis] = face_ext;
            local_vert[t1] = extents[t1] * s1;
            local_vert[t2] = extents[t2] * s2;

            Real rotated[3];
            cudaPhysics::quatRotateVector(rotated, orient, local_vert);
            cudaPhysics::vecAdd3(vertices[i], pos, rotated);
        }
    }

    template <typename Real>
    __device__ int clip_polygon_by_plane(Real output[8][3], const Real input[8][3], int num_input,
                                         const Real *plane_point, const Real *plane_normal, Real epsilon)
    {
        if (num_input == 0)
            return 0;

        int num_output = 0;
        for (int i = 0; i < num_input; ++i)
        {
            int j = (i + 1) % num_input;

            Real d_i = cudaPhysics::dot3(input[i], plane_normal) - cudaPhysics::dot3(plane_point, plane_normal);
            Real d_j = cudaPhysics::dot3(input[j], plane_normal) - cudaPhysics::dot3(plane_point, plane_normal);

            bool inside_i = (d_i <= epsilon);

            if (inside_i)
            {
                cudaPhysics::vecCopy3(output[num_output], input[i]);
                num_output++;
            }

            if ((d_i > epsilon) != (d_j > epsilon))
            {
                Real t = d_i / (d_i - d_j);
                cudaPhysics::axpby(output[num_output], static_cast<Real>(1.0 - t), input[i], t, input[j], 3);
                num_output++;
            }
        }

        return num_output;
    }

    template <typename Real>
    __device__ int generate_face_contacts(
        CollisionInfo<Real> *collisions,
        int *collision_count,
        unsigned int max_collisions,
        int body_id_a, int body_id_b,
        const Real *pos_ref, const Real *orient_ref, const Real *extents_ref,
        const Real *pos_inc, const Real *orient_inc, const Real *extents_inc,
        const Real *orient_ref_conj, const Real *orient_inc_conj,
        const Real axes_ref[3][3],
        int ref_face_idx,
        const Real *n,
        Real epsilon)
    {
        int ref_axis = ref_face_idx % 3;
        int ref_sign = (ref_face_idx < 3) ? 1 : -1;

        Real ref_normal[3];
        cudaPhysics::vecCopy3(ref_normal, axes_ref[ref_axis]);
        if (ref_sign < 0)
            cudaPhysics::vecMul3(ref_normal, static_cast<Real>(-1.0), ref_normal);

        int t1 = (ref_axis + 1) % 3;
        int t2 = (ref_axis + 2) % 3;

        Real ref_face_center[3];
        cudaPhysics::axpby(ref_face_center, static_cast<Real>(1.0), pos_ref, extents_ref[ref_axis], ref_normal, 3);

        Real inc_axes[3][3];
        for (int i = 0; i < 3; ++i)
            get_box_axis(inc_axes[i], orient_inc, i);

        Real min_dot = static_cast<Real>(1e30);
        int inc_face_idx = 0;
        for (int i = 0; i < 6; ++i)
        {
            int axis = i % 3;
            int sign = (i < 3) ? 1 : -1;

            Real inc_normal[3];
            cudaPhysics::vecCopy3(inc_normal, inc_axes[axis]);
            if (sign < 0)
                cudaPhysics::vecMul3(inc_normal, static_cast<Real>(-1.0), inc_normal);

            Real dot_val = cudaPhysics::dot3(ref_normal, inc_normal);
            if (dot_val < min_dot)
            {
                min_dot = dot_val;
                inc_face_idx = i;
            }
        }

        Real inc_verts[8][3];
        get_face_vertices(inc_verts, pos_inc, orient_inc, extents_inc, inc_face_idx);

        Real polygon[8][3];
        int num_verts = 4;
        for (int i = 0; i < 4; ++i)
            cudaPhysics::vecCopy3(polygon[i], inc_verts[i]);

        Real temp[8][3];
        Real side_normals[4][3];
        Real side_points[4][3];

        cudaPhysics::vecCopy3(side_normals[0], axes_ref[t1]);
        cudaPhysics::vecMul3(side_normals[1], (Real)-1, axes_ref[t1]);
        cudaPhysics::vecCopy3(side_normals[2], axes_ref[t2]);
        cudaPhysics::vecMul3(side_normals[3], (Real)-1, axes_ref[t2]);
        cudaPhysics::axpby(side_points[0], (Real)1, ref_face_center, extents_ref[t1], axes_ref[t1], 3);
        cudaPhysics::axpby(side_points[1], (Real)1, ref_face_center, -extents_ref[t1], axes_ref[t1], 3);
        cudaPhysics::axpby(side_points[2], (Real)1, ref_face_center, extents_ref[t2], axes_ref[t2], 3);
        cudaPhysics::axpby(side_points[3], (Real)1, ref_face_center, -extents_ref[t2], axes_ref[t2], 3);

        for (int i = 0; i < 4; ++i)
        {
            num_verts = clip_polygon_by_plane(temp, polygon, num_verts, side_points[i], side_normals[i], epsilon);
            for (int j = 0; j < num_verts; ++j)
                cudaPhysics::vecCopy3(polygon[j], temp[j]);
        }

        int num_contacts = 0;
        for (int i = 0; i < num_verts && num_contacts < MAX_MANIFOLD_POINTS; ++i)
        {
            Real penetration = cudaPhysics::dot3(ref_normal, ref_face_center) - cudaPhysics::dot3(ref_normal, polygon[i]);

            if (penetration >= 0)
            {
                Real contact_a[3], contact_b[3];

                Real projected[3];
                cudaPhysics::axpby(projected, static_cast<Real>(1.0), polygon[i], penetration, ref_normal, 3);

                cudaPhysics::get_local_point(contact_a, pos_ref, orient_ref_conj, projected);
                cudaPhysics::get_local_point(contact_b, pos_inc, orient_inc_conj, polygon[i]);

                num_contacts += CollisionDataUtils::add_collision_info(collisions, collision_count, max_collisions, body_id_a, body_id_b, contact_a, contact_b, ref_normal);
            }
        }

        return num_contacts;
    }

    template <typename Real>
    __device__ int generate_edge_contacts(
        CollisionInfo<Real> *collisions,
        int *collision_count,
        unsigned int max_collisions,
        int body_id_a, int body_id_b,
        const Real *pos_a, const Real *orient_a, const Real *extents_a,
        const Real *pos_b, const Real *orient_b, const Real *extents_b,
        const Real *orient_a_conj, const Real *orient_b_conj,
        int edge_axis_a, int edge_axis_b,
        const Real *n,
        Real epsilon)
    {
        Real axes_a[3][3], axes_b[3][3];

        for (int i = 0; i < 3; i++)
        {
            get_box_axis(axes_a[i], orient_a, i);
            get_box_axis(axes_b[i], orient_b, i);
        }

        Real dA[3], dB[3];
        cudaPhysics::vecCopy3(dA, axes_a[edge_axis_a]);
        cudaPhysics::vecCopy3(dB, axes_b[edge_axis_b]);

        int t1 = (edge_axis_a + 1) % 3;
        int t2 = (edge_axis_a + 2) % 3;

        int u1 = (edge_axis_b + 1) % 3;
        int u2 = (edge_axis_b + 2) % 3;

        Real pA[3], pB[3];
        Real localA[3] = {0, 0, 0};
        Real localB[3] = {0, 0, 0};
        Real sA1 = cudaPhysics::dot3(n, axes_a[t1]) > 0 ? 1 : -1;
        Real sA2 = cudaPhysics::dot3(n, axes_a[t2]) > 0 ? 1 : -1;
        Real sB1 = cudaPhysics::dot3(n, axes_b[u1]) > 0 ? -1 : 1;
        Real sB2 = cudaPhysics::dot3(n, axes_b[u2]) > 0 ? -1 : 1;
        localA[t1] = sA1 * extents_a[t1];
        localA[t2] = sA2 * extents_a[t2];
        localB[u1] = sB1 * extents_b[u1];
        localB[u2] = sB2 * extents_b[u2];

        cudaPhysics::quatRotateVector(pA, orient_a, localA);
        cudaPhysics::vecAdd3(pA, pos_a, pA);

        cudaPhysics::quatRotateVector(pB, orient_b, localB);
        cudaPhysics::vecAdd3(pB, pos_b, pB);

        Real r[3];
        cudaPhysics::vecSubs3(r, pA, pB);

        Real a = cudaPhysics::dot3(dA, dA);
        Real e = cudaPhysics::dot3(dB, dB);
        Real b = cudaPhysics::dot3(dA, dB);
        Real c = cudaPhysics::dot3(dA, r);
        Real f = cudaPhysics::dot3(dB, r);

        Real denom = a * e - b * b;

        if (fabs(denom) < epsilon)
            return 0;

        Real t = (b * f - c * e) / denom;
        Real s = (a * f - b * c) / denom;

        Real extentA = extents_a[edge_axis_a];
        Real extentB = extents_b[edge_axis_b];

        t = max(-extentA, min(extentA, t));
        s = max(-extentB, min(extentB, s));

        Real cpA[3], cpB[3];

        cudaPhysics::axpby(cpA, (Real)1.0, pA, t, dA, 3);
        cudaPhysics::axpby(cpB, (Real)1.0, pB, s, dB, 3);

        Real contact_world[3];
        cudaPhysics::axpby(contact_world, (Real)0.5, cpA, (Real)0.5, cpB, 3);

        Real contact_a[3], contact_b[3];

        cudaPhysics::get_local_point(contact_a, pos_a, orient_a_conj, contact_world);
        cudaPhysics::get_local_point(contact_b, pos_b, orient_b_conj, contact_world);

        return CollisionDataUtils::add_collision_info(
            collisions, collision_count, max_collisions,
            body_id_a, body_id_b,
            contact_a, contact_b,
            n);
    }

    template <typename Real>
    __device__ int collide_box_box(
        CollisionInfo<Real> *collisions,
        int *collision_count,
        unsigned int max_collisions,
        int body_id_a, int body_id_b,
        const Real *pos_a, const Real *orient_a, Real hx_a, Real hy_a, Real hz_a,
        const Real *pos_b, const Real *orient_b, Real hx_b, Real hy_b, Real hz_b,
        Real epsilon)
    {
        Real axes_a[3][3], axes_b[3][3];
        for (int i = 0; i < 3; ++i)
        {
            get_box_axis(axes_a[i], orient_a, i);
            get_box_axis(axes_b[i], orient_b, i);
        }

        Real min_overlap = static_cast<Real>(1e30);
        Real min_axis[3];
        int min_axis_type = 0;
        int min_axis_idx = 0;
        int min_axis_idx_j = 0;

        for (int i = 0; i < 3; ++i)
        {
            Real overlap;
            if (!test_axis(overlap, axes_a[i], pos_a, orient_a, hx_a, hy_a, hz_a, pos_b, orient_b, hx_b, hy_b, hz_b, epsilon))
                return 0;

            if (overlap < min_overlap)
            {
                min_overlap = overlap;
                cudaPhysics::vecCopy3(min_axis, axes_a[i]);
                min_axis_type = 0;
                min_axis_idx = i;
            }
        }

        for (int i = 0; i < 3; ++i)
        {
            Real overlap;
            if (!test_axis(overlap, axes_b[i], pos_a, orient_a, hx_a, hy_a, hz_a, pos_b, orient_b, hx_b, hy_b, hz_b, epsilon))
                return 0;

            if (overlap < min_overlap)
            {
                min_overlap = overlap;
                cudaPhysics::vecCopy3(min_axis, axes_b[i]);
                min_axis_type = 1;
                min_axis_idx = i;
            }
        }

        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                Real cross_axis[3];
                cudaPhysics::cross3(cross_axis, axes_a[i], axes_b[j]);

                Real cross_len = cudaPhysics::len3(cross_axis);
                if (cross_len < epsilon)
                    continue;

                Real overlap;
                if (!test_axis(overlap, cross_axis, pos_a, orient_a, hx_a, hy_a, hz_a, pos_b, orient_b, hx_b, hy_b, hz_b, epsilon))
                    return 0;

                if (overlap < min_overlap)
                {
                    min_overlap = overlap;
                    cudaPhysics::vecCopy3(min_axis, cross_axis);
                    min_axis_type = 2;
                    min_axis_idx = i;
                    min_axis_idx_j = j;
                }
            }
        }

        if (min_overlap <= epsilon)
            return 0;

        Real d[3];
        cudaPhysics::vecSubs3(d, pos_b, pos_a);
        Real sign = cudaPhysics::dot3(d, min_axis) < 0.0 ? -1.0 : 1.0;
        cudaPhysics::vecMul3(min_axis, sign, min_axis);

        Real len = cudaPhysics::len3(min_axis);
        if (len < epsilon)
            return 0;

        Real n[3];
        cudaPhysics::vecMul3(n, static_cast<Real>(1.0) / len, min_axis);

        Real orient_a_conj[4], orient_b_conj[4];
        cudaPhysics::quatConjugate(orient_a_conj, orient_a);
        cudaPhysics::quatConjugate(orient_b_conj, orient_b);

        Real extents_a[3] = {hx_a, hy_a, hz_a};
        Real extents_b[3] = {hx_b, hy_b, hz_b};

        int num_contacts = 0;

        if (min_axis_type == 2)
        {
            num_contacts = generate_edge_contacts(
                collisions, collision_count, max_collisions,
                body_id_a, body_id_b,
                pos_a, orient_a, extents_a,
                pos_b, orient_b, extents_b,
                orient_a_conj, orient_b_conj,
                min_axis_idx, min_axis_idx_j,
                n, epsilon);
        }
        else
        {
            int input_a_id, input_b_id;
            const Real *pos_ref, *pos_inc;
            const Real *orient_ref, *orient_inc;
            const Real *extents_ref, *extents_inc;
            const Real *orient_ref_conj, *orient_inc_conj;
            const Real(*axes_ref)[3];
            int ref_face_idx;

            if (min_axis_type == 0)
            {
                input_a_id = body_id_a;
                input_b_id = body_id_b;
                pos_ref = pos_a;
                pos_inc = pos_b;
                orient_ref = orient_a;
                orient_inc = orient_b;
                extents_ref = extents_a;
                extents_inc = extents_b;
                orient_ref_conj = orient_a_conj;
                orient_inc_conj = orient_b_conj;
                axes_ref = axes_a;
                ref_face_idx = min_axis_idx + (sign > 0.0 ? 0 : 3);
            }
            else
            {
                input_a_id = body_id_b;
                input_b_id = body_id_a;
                pos_ref = pos_b;
                pos_inc = pos_a;
                orient_ref = orient_b;
                orient_inc = orient_a;
                extents_ref = extents_b;
                extents_inc = extents_a;
                orient_ref_conj = orient_b_conj;
                orient_inc_conj = orient_a_conj;
                axes_ref = axes_b;
                ref_face_idx = min_axis_idx + (sign < 0.0 ? 0 : 3);
            }

            num_contacts = generate_face_contacts(
                collisions, collision_count, max_collisions,
                input_a_id, input_b_id,
                pos_ref, orient_ref, extents_ref,
                pos_inc, orient_inc, extents_inc,
                orient_ref_conj, orient_inc_conj,
                axes_ref, ref_face_idx, n, epsilon);
        }

        return num_contacts;
    }
}

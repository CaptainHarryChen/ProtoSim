#pragma once
#include <Math/algebra.cuh>
#include <collision/CollisionDataUtils.cuh>

namespace BodyCollisionKernel
{
    template <typename Real>
    __device__ void get_box_axis(Real *axis, const Real *orient, int axis_idx)
    {
        if (axis_idx == 0)
        {
            axis[0] = static_cast<Real>(1.0);
            axis[1] = static_cast<Real>(0.0);
            axis[2] = static_cast<Real>(0.0);
        }
        else if (axis_idx == 1)
        {
            axis[0] = static_cast<Real>(0.0);
            axis[1] = static_cast<Real>(1.0);
            axis[2] = static_cast<Real>(0.0);
        }
        else
        {
            axis[0] = static_cast<Real>(0.0);
            axis[1] = static_cast<Real>(0.0);
            axis[2] = static_cast<Real>(1.0);
        }
        cudaPhysics::quatRotateVector(axis, orient, axis);
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
    __device__ void get_box_face_vertices(Real vertices[4][3], const Real *pos, const Real *orient, Real hx, Real hy, Real hz, int face_idx)
    {
        int sign = (face_idx >= 3) ? -1 : 1;
        int axis = face_idx % 3;

        Real local_verts[4][3];
        if (axis == 0)
        {
            local_verts[0][0] = sign * hx;
            local_verts[0][1] = -hy;
            local_verts[0][2] = -hz;
            local_verts[1][0] = sign * hx;
            local_verts[1][1] = hy;
            local_verts[1][2] = -hz;
            local_verts[2][0] = sign * hx;
            local_verts[2][1] = hy;
            local_verts[2][2] = hz;
            local_verts[3][0] = sign * hx;
            local_verts[3][1] = -hy;
            local_verts[3][2] = hz;
        }
        else if (axis == 1)
        {
            local_verts[0][0] = -hx;
            local_verts[0][1] = sign * hy;
            local_verts[0][2] = -hz;
            local_verts[1][0] = hx;
            local_verts[1][1] = sign * hy;
            local_verts[1][2] = -hz;
            local_verts[2][0] = hx;
            local_verts[2][1] = sign * hy;
            local_verts[2][2] = hz;
            local_verts[3][0] = -hx;
            local_verts[3][1] = sign * hy;
            local_verts[3][2] = hz;
        }
        else
        {
            local_verts[0][0] = -hx;
            local_verts[0][1] = -hy;
            local_verts[0][2] = sign * hz;
            local_verts[1][0] = hx;
            local_verts[1][1] = -hy;
            local_verts[1][2] = sign * hz;
            local_verts[2][0] = hx;
            local_verts[2][1] = hy;
            local_verts[2][2] = sign * hz;
            local_verts[3][0] = -hx;
            local_verts[3][1] = hy;
            local_verts[3][2] = sign * hz;
        }

        for (int i = 0; i < 4; ++i)
        {
            Real rotated[3];
            cudaPhysics::quatRotateVector(rotated, orient, local_verts[i]);
            cudaPhysics::vecAdd3(vertices[i], pos, rotated);
        }
    }

    template <typename Real>
    __device__ Real distance_point_to_plane(const Real *point, const Real *plane_point, const Real *plane_normal)
    {
        Real diff[3];
        cudaPhysics::vecSubs3(diff, point, plane_point);
        return cudaPhysics::dot3(diff, plane_normal);
    }

    template <typename Real>
    __device__ void project_point_onto_plane(Real *projected, const Real *point, const Real *plane_point, const Real *plane_normal)
    {
        Real dist = distance_point_to_plane(point, plane_point, plane_normal);
        projected[0] = point[0] - dist * plane_normal[0];
        projected[1] = point[1] - dist * plane_normal[1];
        projected[2] = point[2] - dist * plane_normal[2];
    }

    template <typename Real>
    __device__ int clip_polygon_by_plane(Real output[8][3], const Real input[8][3], int num_input, const Real *plane_point, const Real *plane_normal, Real epsilon)
    {
        if (num_input == 0)
            return 0;

        Real distances[8];
        bool inside[8];

        for (int i = 0; i < num_input; ++i)
        {
            distances[i] = distance_point_to_plane(input[i], plane_point, plane_normal);
            inside[i] = (distances[i] >= -epsilon);
        }

        int num_output = 0;
        for (int i = 0; i < num_input; ++i)
        {
            int j = (i + 1) % num_input;

            if (inside[i])
            {
                cudaPhysics::vecCopy3(output[num_output], input[i]);
                num_output++;
            }

            if (inside[i] != inside[j])
            {
                Real t = distances[i] / (distances[i] - distances[j]);
                output[num_output][0] = input[i][0] + t * (input[j][0] - input[i][0]);
                output[num_output][1] = input[i][1] + t * (input[j][1] - input[i][1]);
                output[num_output][2] = input[i][2] + t * (input[j][2] - input[i][2]);
                num_output++;
            }
        }

        return num_output;
    }

    template <typename Real>
    __device__ int clip_polygon_by_box_face(Real output[8][3], const Real input[8][3], int num_input,
                                            const Real *face_center, const Real *face_tangent1, const Real *face_tangent2,
                                            Real extent1, Real extent2, Real epsilon)
    {
        Real temp1[8][3], temp2[8][3];
        int num_temp1 = num_input;

        for (int i = 0; i < num_input; ++i)
            cudaPhysics::vecCopy3(temp1[i], input[i]);

        for (int edge = 0; edge < 4; ++edge)
        {
            if (num_temp1 == 0)
                return 0;

            Real edge_point[3], edge_normal[3];
            int sign = (edge < 2) ? 1 : -1;
            int axis = edge % 2;

            if (axis == 0)
            {
                edge_point[0] = face_center[0] + sign * extent1 * face_tangent1[0];
                edge_point[1] = face_center[1] + sign * extent1 * face_tangent1[1];
                edge_point[2] = face_center[2] + sign * extent1 * face_tangent1[2];

                cudaPhysics::cross3(edge_normal, face_tangent1, face_tangent2);
                if (sign < 0)
                    cudaPhysics::vecMul3(edge_normal, static_cast<Real>(-1.0), edge_normal);
            }
            else
            {
                edge_point[0] = face_center[0] + sign * extent2 * face_tangent2[0];
                edge_point[1] = face_center[1] + sign * extent2 * face_tangent2[1];
                edge_point[2] = face_center[2] + sign * extent2 * face_tangent2[2];

                cudaPhysics::cross3(edge_normal, face_tangent2, face_tangent1);
                if (sign < 0)
                    cudaPhysics::vecMul3(edge_normal, static_cast<Real>(-1.0), edge_normal);
            }

            num_temp1 = clip_polygon_by_plane(temp2, temp1, num_temp1, edge_point, edge_normal, epsilon);

            for (int i = 0; i < num_temp1; ++i)
                cudaPhysics::vecCopy3(temp1[i], temp2[i]);
        }

        for (int i = 0; i < num_temp1; ++i)
            cudaPhysics::vecCopy3(output[i], temp1[i]);

        return num_temp1;
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
                }
            }
        }

        Real d[3];
        cudaPhysics::vecSubs3(d, pos_b, pos_a);
        if (cudaPhysics::dot3(d, min_axis) < static_cast<Real>(0.0))
        {
            min_axis[0] = -min_axis[0];
            min_axis[1] = -min_axis[1];
            min_axis[2] = -min_axis[2];
        }

        Real len = cudaPhysics::len3(min_axis);
        if (len < epsilon)
            return 0;

        Real n[3];
        cudaPhysics::vecMul3(n, static_cast<Real>(1.0) / len, min_axis);

        Real orient_a_conj[4], orient_b_conj[4];
        cudaPhysics::quatConjugate(orient_a_conj, orient_a);
        cudaPhysics::quatConjugate(orient_b_conj, orient_b);

        int num_contacts = 0;

        if (min_axis_type == 0)
        {
            int face_idx = min_axis_idx;
            Real face_center[3];
            Real face_normal[3];
            Real face_tangent1[3], face_tangent2[3];
            Real extent1, extent2;

            cudaPhysics::vecCopy3(face_normal, axes_a[face_idx]);
            Real sign_n = (cudaPhysics::dot3(n, face_normal) > 0) ? static_cast<Real>(1.0) : static_cast<Real>(-1.0);
            cudaPhysics::vecMul3(face_normal, sign_n, face_normal);

            int tangent_idx1 = (face_idx + 1) % 3;
            int tangent_idx2 = (face_idx + 2) % 3;
            cudaPhysics::vecCopy3(face_tangent1, axes_a[tangent_idx1]);
            cudaPhysics::vecCopy3(face_tangent2, axes_a[tangent_idx2]);

            if (face_idx == 0)
            {
                extent1 = hy_a;
                extent2 = hz_a;
                face_center[0] = pos_a[0] + sign_n * hx_a * face_normal[0];
                face_center[1] = pos_a[1] + sign_n * hx_a * face_normal[1];
                face_center[2] = pos_a[2] + sign_n * hx_a * face_normal[2];
            }
            else if (face_idx == 1)
            {
                extent1 = hx_a;
                extent2 = hz_a;
                face_center[0] = pos_a[0] + sign_n * hy_a * face_normal[0];
                face_center[1] = pos_a[1] + sign_n * hy_a * face_normal[1];
                face_center[2] = pos_a[2] + sign_n * hy_a * face_normal[2];
            }
            else
            {
                extent1 = hx_a;
                extent2 = hy_a;
                face_center[0] = pos_a[0] + sign_n * hz_a * face_normal[0];
                face_center[1] = pos_a[1] + sign_n * hz_a * face_normal[1];
                face_center[2] = pos_a[2] + sign_n * hz_a * face_normal[2];
            }

            Real incident_verts[8][3];
            get_box_face_vertices(incident_verts, pos_b, orient_b, hx_b, hy_b, hz_b, 0);

            Real best_verts[8][3];
            Real best_dist = static_cast<Real>(1e30);
            for (int face = 0; face < 6; ++face)
            {
                Real verts[8][3];
                get_box_face_vertices(verts, pos_b, orient_b, hx_b, hy_b, hz_b, face);

                Real avg_dist = static_cast<Real>(0.0);
                for (int v = 0; v < 4; ++v)
                {
                    avg_dist += distance_point_to_plane(verts[v], face_center, face_normal);
                }
                avg_dist /= static_cast<Real>(4.0);

                if (abs(avg_dist) < abs(best_dist))
                {
                    best_dist = avg_dist;
                    for (int v = 0; v < 4; ++v)
                        cudaPhysics::vecCopy3(best_verts[v], verts[v]);
                }
            }

            Real clipped[8][3];
            int num_clipped = clip_polygon_by_box_face(clipped, best_verts, 4, face_center, face_tangent1, face_tangent2, extent1, extent2, epsilon);

            for (int i = 0; i < num_clipped && num_contacts < MAX_MANIFOLD_POINTS; ++i)
            {
                Real dist = distance_point_to_plane(clipped[i], face_center, face_normal);
                if (dist < static_cast<Real>(0.0))
                {
                    Real contact_a[3], contact_b[3];

                    Real projected[3];
                    project_point_onto_plane(projected, clipped[i], face_center, face_normal);
                    cudaPhysics::get_local_point(contact_a, pos_a, orient_a_conj, projected);
                    cudaPhysics::get_local_point(contact_b, pos_b, orient_b_conj, clipped[i]);

                    Real contact_normal[3];
                    cudaPhysics::vecMul3(contact_normal, static_cast<Real>(-1.0), face_normal);

                    num_contacts += CollisionDataUtils::add_collision_info(collisions, collision_count, max_collisions, body_id_a, body_id_b, contact_a, contact_b, contact_normal);
                }
            }
        }
        else if (min_axis_type == 1)
        {
            int face_idx = min_axis_idx;
            Real face_center[3];
            Real face_normal[3];
            Real face_tangent1[3], face_tangent2[3];
            Real extent1, extent2;

            cudaPhysics::vecCopy3(face_normal, axes_b[face_idx]);
            Real sign_n = (cudaPhysics::dot3(n, face_normal) < 0) ? static_cast<Real>(1.0) : static_cast<Real>(-1.0);
            cudaPhysics::vecMul3(face_normal, sign_n, face_normal);

            int tangent_idx1 = (face_idx + 1) % 3;
            int tangent_idx2 = (face_idx + 2) % 3;
            cudaPhysics::vecCopy3(face_tangent1, axes_b[tangent_idx1]);
            cudaPhysics::vecCopy3(face_tangent2, axes_b[tangent_idx2]);

            if (face_idx == 0)
            {
                extent1 = hy_b;
                extent2 = hz_b;
                face_center[0] = pos_b[0] + sign_n * hx_b * face_normal[0];
                face_center[1] = pos_b[1] + sign_n * hx_b * face_normal[1];
                face_center[2] = pos_b[2] + sign_n * hx_b * face_normal[2];
            }
            else if (face_idx == 1)
            {
                extent1 = hx_b;
                extent2 = hz_b;
                face_center[0] = pos_b[0] + sign_n * hy_b * face_normal[0];
                face_center[1] = pos_b[1] + sign_n * hy_b * face_normal[1];
                face_center[2] = pos_b[2] + sign_n * hy_b * face_normal[2];
            }
            else
            {
                extent1 = hx_b;
                extent2 = hy_b;
                face_center[0] = pos_b[0] + sign_n * hz_b * face_normal[0];
                face_center[1] = pos_b[1] + sign_n * hz_b * face_normal[1];
                face_center[2] = pos_b[2] + sign_n * hz_b * face_normal[2];
            }

            Real incident_verts[8][3];
            get_box_face_vertices(incident_verts, pos_a, orient_a, hx_a, hy_a, hz_a, 0);

            Real best_verts[8][3];
            Real best_dist = static_cast<Real>(1e30);
            for (int face = 0; face < 6; ++face)
            {
                Real verts[8][3];
                get_box_face_vertices(verts, pos_a, orient_a, hx_a, hy_a, hz_a, face);

                Real avg_dist = static_cast<Real>(0.0);
                for (int v = 0; v < 4; ++v)
                {
                    avg_dist += distance_point_to_plane(verts[v], face_center, face_normal);
                }
                avg_dist /= static_cast<Real>(4.0);

                if (abs(avg_dist) < abs(best_dist))
                {
                    best_dist = avg_dist;
                    for (int v = 0; v < 4; ++v)
                        cudaPhysics::vecCopy3(best_verts[v], verts[v]);
                }
            }

            Real clipped[8][3];
            int num_clipped = clip_polygon_by_box_face(clipped, best_verts, 4, face_center, face_tangent1, face_tangent2, extent1, extent2, epsilon);

            for (int i = 0; i < num_clipped && num_contacts < MAX_MANIFOLD_POINTS; ++i)
            {
                Real dist = distance_point_to_plane(clipped[i], face_center, face_normal);
                if (dist < static_cast<Real>(0.0))
                {
                    Real contact_a[3], contact_b[3];

                    cudaPhysics::get_local_point(contact_a, pos_a, orient_a_conj, clipped[i]);

                    Real projected[3];
                    project_point_onto_plane(projected, clipped[i], face_center, face_normal);
                    cudaPhysics::get_local_point(contact_b, pos_b, orient_b_conj, projected);

                    num_contacts += CollisionDataUtils::add_collision_info(collisions, collision_count, max_collisions, body_id_a, body_id_b, contact_a, contact_b, n);
                }
            }
        }
        else
        {
            Real support_a[3], support_b[3];

            Real n_local_a[3], n_local_b[3];
            cudaPhysics::quatRotateVector(n_local_a, orient_a_conj, n);
            cudaPhysics::quatRotateVector(n_local_b, orient_b_conj, n);

            support_a[0] = n_local_a[0] > 0 ? hx_a : -hx_a;
            support_a[1] = n_local_a[1] > 0 ? hy_a : -hy_a;
            support_a[2] = n_local_a[2] > 0 ? hz_a : -hz_a;

            support_b[0] = n_local_b[0] < 0 ? hx_b : -hx_b;
            support_b[1] = n_local_b[1] < 0 ? hy_b : -hy_b;
            support_b[2] = n_local_b[2] < 0 ? hz_b : -hz_b;

            Real world_support_a[3], world_support_b[3];
            cudaPhysics::quatRotateVector(world_support_a, orient_a, support_a);
            cudaPhysics::vecAdd3(world_support_a, pos_a, world_support_a);
            cudaPhysics::quatRotateVector(world_support_b, orient_b, support_b);
            cudaPhysics::vecAdd3(world_support_b, pos_b, world_support_b);

            Real contact_a[3], contact_b[3];
            cudaPhysics::get_local_point(contact_a, pos_a, orient_a_conj, world_support_a);
            cudaPhysics::get_local_point(contact_b, pos_b, orient_b_conj, world_support_b);

            num_contacts = CollisionDataUtils::add_collision_info(collisions, collision_count, max_collisions, body_id_a, body_id_b, contact_a, contact_b, n);
        }

        return num_contacts;
    }
}

#include "BodyCollision.cuh"
#include <cuda_utils/cuda_utils.cuh>
#include <Math/algebra.cuh>

#define EPSILON 1e-6

namespace BodyCollisionKernel
{
    template <typename Real>
    __device__ void get_world_point(Real* world_point, const Real* pos, const Real* orient, const Real* local_point)
    {
        Real rotated[3];
        cudaPhysics::quatRotateVector(rotated, orient, local_point);
        cudaPhysics::vecAdd3(world_point, pos, rotated);
    }

    template <typename Real>
    __device__ void get_local_point(Real* local_point, const Real* pos, const Real* orient_conj, const Real* world_point)
    {
        Real r[3];
        cudaPhysics::vecSubs3(r, world_point, pos);
        cudaPhysics::quatRotateVector(local_point, orient_conj, r);
    }

    template <typename Real>
    __device__ bool collide_sphere_sphere(
        Real* contact_a, Real* contact_b, Real* normal, Real& penetration,
        const Real* pos_a, const Real* orient_a, Real radius_a,
        const Real* pos_b, const Real* orient_b, Real radius_b)
    {
        Real diff[3];
        cudaPhysics::vecSubs3(diff, pos_b, pos_a);
        Real dist = cudaPhysics::len3(diff);

        Real sum_radius = radius_a + radius_b;

        if (dist >= sum_radius || dist < EPSILON)
            return false;

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

        get_local_point(contact_a, pos_a, orient_a_conj, contact_on_a);
        get_local_point(contact_b, pos_b, orient_b_conj, contact_on_b);

        normal[0] = n[0];
        normal[1] = n[1];
        normal[2] = n[2];

        penetration = sum_radius - dist;

        return true;
    }

    template <typename Real>
    __device__ void clamp_to_box(Real* clamped, const Real* point, Real hx, Real hy, Real hz)
    {
        clamped[0] = max(-hx, min(hx, point[0]));
        clamped[1] = max(-hy, min(hy, point[1]));
        clamped[2] = max(-hz, min(hz, point[2]));
    }

    template <typename Real>
    __device__ bool collide_sphere_box(
        Real* contact_a, Real* contact_b, Real* normal, Real& penetration,
        const Real* pos_sphere, const Real* orient_sphere, Real radius,
        const Real* pos_box, const Real* orient_box, Real hx, Real hy, Real hz)
    {
        Real orient_box_conj[4];
        cudaPhysics::quatConjugate(orient_box_conj, orient_box);

        Real sphere_center_local[3];
        Real diff[3];
        cudaPhysics::vecSubs3(diff, pos_sphere, pos_box);
        cudaPhysics::quatRotateVector(sphere_center_local, orient_box_conj, diff);

        Real closest_local[3];
        clamp_to_box(closest_local, sphere_center_local, hx, hy, hz);

        Real diff_local[3];
        cudaPhysics::vecSubs3(diff_local, sphere_center_local, closest_local);
        Real dist = cudaPhysics::len3(diff_local);

        if (dist >= radius)
            return false;

        Real n_local[3];
        if (dist > EPSILON)
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
        cudaPhysics::vecMul3(contact_on_sphere, -radius, n_world);
        cudaPhysics::vecAdd3(contact_on_sphere, pos_sphere, contact_on_sphere);

        Real contact_on_box[3];
        cudaPhysics::quatRotateVector(contact_on_box, orient_box, closest_local);
        cudaPhysics::vecAdd3(contact_on_box, pos_box, contact_on_box);

        Real orient_sphere_conj[4];
        cudaPhysics::quatConjugate(orient_sphere_conj, orient_sphere);

        get_local_point(contact_a, pos_sphere, orient_sphere_conj, contact_on_sphere);
        get_local_point(contact_b, pos_box, orient_box_conj, contact_on_box);

        normal[0] = -n_world[0];
        normal[1] = -n_world[1];
        normal[2] = -n_world[2];

        penetration = radius - dist;

        return true;
    }

    template <typename Real>
    __device__ void get_box_axis(Real* axis, const Real* orient, int axis_idx)
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
    __device__ void project_box_on_axis(Real& min_proj, Real& max_proj, const Real* pos, const Real* orient, Real hx, Real hy, Real hz, const Real* axis)
    {
        Real corners[8][3] = {
            {-hx, -hy, -hz}, {hx, -hy, -hz}, {-hx, hy, -hz}, {hx, hy, -hz},
            {-hx, -hy, hz}, {hx, -hy, hz}, {-hx, hy, hz}, {hx, hy, hz}};

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
    __device__ bool test_axis(Real& overlap, const Real* axis,
        const Real* pos_a, const Real* orient_a, Real hx_a, Real hy_a, Real hz_a,
        const Real* pos_b, const Real* orient_b, Real hx_b, Real hy_b, Real hz_b)
    {
        Real len = cudaPhysics::len3(axis);
        if (len < EPSILON)
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
    __device__ bool collide_box_box(
        Real* contact_a, Real* contact_b, Real* normal, Real& penetration,
        const Real* pos_a, const Real* orient_a, Real hx_a, Real hy_a, Real hz_a,
        const Real* pos_b, const Real* orient_b, Real hx_b, Real hy_b, Real hz_b)
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

        for (int i = 0; i < 3; ++i)
        {
            Real overlap;
            if (!test_axis(overlap, axes_a[i], pos_a, orient_a, hx_a, hy_a, hz_a, pos_b, orient_b, hx_b, hy_b, hz_b))
                return false;

            if (overlap < min_overlap)
            {
                min_overlap = overlap;
                cudaPhysics::vecCopy3(min_axis, axes_a[i]);
                min_axis_type = 0;
            }
        }

        for (int i = 0; i < 3; ++i)
        {
            Real overlap;
            if (!test_axis(overlap, axes_b[i], pos_a, orient_a, hx_a, hy_a, hz_a, pos_b, orient_b, hx_b, hy_b, hz_b))
                return false;

            if (overlap < min_overlap)
            {
                min_overlap = overlap;
                cudaPhysics::vecCopy3(min_axis, axes_b[i]);
                min_axis_type = 1;
            }
        }

        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                Real cross_axis[3];
                cudaPhysics::cross3(cross_axis, axes_a[i], axes_b[j]);

                Real overlap;
                if (!test_axis(overlap, cross_axis, pos_a, orient_a, hx_a, hy_a, hz_a, pos_b, orient_b, hx_b, hy_b, hz_b))
                    return false;

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
        if (len < EPSILON)
            return false;

        Real n[3];
        cudaPhysics::vecMul3(n, static_cast<Real>(1.0) / len, min_axis);

        Real center_a[3] = {0, 0, 0};
        Real center_b[3] = {0, 0, 0};

        Real orient_a_conj[4], orient_b_conj[4];
        cudaPhysics::quatConjugate(orient_a_conj, orient_a);
        cudaPhysics::quatConjugate(orient_b_conj, orient_b);

        Real n_local_a[3], n_local_b[3];
        cudaPhysics::quatRotateVector(n_local_a, orient_a_conj, n);
        cudaPhysics::quatRotateVector(n_local_b, orient_b_conj, n);

        Real support_a[3];
        support_a[0] = n_local_a[0] > 0 ? hx_a : -hx_a;
        support_a[1] = n_local_a[1] > 0 ? hy_a : -hy_a;
        support_a[2] = n_local_a[2] > 0 ? hz_a : -hz_a;

        Real support_b[3];
        support_b[0] = n_local_b[0] < 0 ? hx_b : -hx_b;
        support_b[1] = n_local_b[1] < 0 ? hy_b : -hy_b;
        support_b[2] = n_local_b[2] < 0 ? hz_b : -hz_b;

        Real world_support_a[3], world_support_b[3];
        cudaPhysics::quatRotateVector(world_support_a, orient_a, support_a);
        cudaPhysics::vecAdd3(world_support_a, pos_a, world_support_a);
        cudaPhysics::quatRotateVector(world_support_b, orient_b, support_b);
        cudaPhysics::vecAdd3(world_support_b, pos_b, world_support_b);

        Real contact_world[3];
        cudaPhysics::vecAdd3(contact_world, world_support_a, world_support_b);
        cudaPhysics::vecMul3(contact_world, static_cast<Real>(0.5), contact_world);

        get_local_point(contact_a, pos_a, orient_a_conj, contact_world);
        get_local_point(contact_b, pos_b, orient_b_conj, contact_world);

        normal[0] = n[0];
        normal[1] = n[1];
        normal[2] = n[2];

        penetration = min_overlap;

        return true;
    }

    template <typename Real>
    __device__ bool collide_capsule_sphere(
        Real* contact_a, Real* contact_b, Real* normal, Real& penetration,
        const Real* pos_capsule, const Real* orient_capsule, Real radius_capsule, Real half_height,
        const Real* pos_sphere, const Real* orient_sphere, Real radius_sphere)
    {
        Real axis[3] = {0, 1, 0};
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

            if (ab_len_sq < EPSILON)
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

        if (dist >= sum_radius || dist < EPSILON)
            return false;

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

        get_local_point(contact_a, pos_capsule, orient_capsule_conj, contact_on_capsule);
        get_local_point(contact_b, pos_sphere, orient_sphere_conj, contact_on_sphere);

        normal[0] = n[0];
        normal[1] = n[1];
        normal[2] = n[2];

        penetration = sum_radius - dist;

        return true;
    }

    template <typename Real>
    __device__ bool collide_capsule_box(
        Real* contact_a, Real* contact_b, Real* normal, Real& penetration,
        const Real* pos_capsule, const Real* orient_capsule, Real radius_capsule, Real half_height,
        const Real* pos_box, const Real* orient_box, Real hx, Real hy, Real hz)
    {
        Real axis[3] = {0, 1, 0};
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

        Real best_dist = static_cast<Real>(1e30);
        Real best_point_local[3];
        Real best_t = static_cast<Real>(0.0);

        Real ab_local[3];
        cudaPhysics::vecSubs3(ab_local, top_local, bottom_local);
        Real ab_len_sq = cudaPhysics::dot3(ab_local, ab_local);

        int num_samples = 16;
        for (int i = 0; i <= num_samples; ++i)
        {
            Real t = static_cast<Real>(i) / static_cast<Real>(num_samples);
            Real point_local[3];
            cudaPhysics::axpby(point_local, static_cast<Real>(1.0) - t, bottom_local, t, top_local, 3);

            Real clamped[3];
            clamp_to_box(clamped, point_local, hx, hy, hz);

            Real diff[3];
            cudaPhysics::vecSubs3(diff, point_local, clamped);
            Real dist = cudaPhysics::len3(diff);

            if (dist < best_dist)
            {
                best_dist = dist;
                cudaPhysics::vecCopy3(best_point_local, clamped);
                best_t = t;
            }
        }

        if (best_dist >= radius_capsule)
            return false;

        Real closest_on_segment_local[3];
        cudaPhysics::axpby(closest_on_segment_local, static_cast<Real>(1.0) - best_t, bottom_local, best_t, top_local, 3);

        Real n_local[3];
        if (best_dist > EPSILON)
        {
            Real diff[3];
            cudaPhysics::vecSubs3(diff, closest_on_segment_local, best_point_local);
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
                n_local[0] = closest_on_segment_local[0] > 0 ? static_cast<Real>(1.0) : static_cast<Real>(-1.0);
                n_local[1] = static_cast<Real>(0.0);
                n_local[2] = static_cast<Real>(0.0);
            }
            else if (abs_y <= abs_z)
            {
                n_local[0] = static_cast<Real>(0.0);
                n_local[1] = closest_on_segment_local[1] > 0 ? static_cast<Real>(1.0) : static_cast<Real>(-1.0);
                n_local[2] = static_cast<Real>(0.0);
            }
            else
            {
                n_local[0] = static_cast<Real>(0.0);
                n_local[1] = static_cast<Real>(0.0);
                n_local[2] = closest_on_segment_local[2] > 0 ? static_cast<Real>(1.0) : static_cast<Real>(-1.0);
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

        get_local_point(contact_a, pos_capsule, orient_capsule_conj, contact_on_capsule);
        get_local_point(contact_b, pos_box, orient_box_conj, contact_on_box);

        normal[0] = -n_world[0];
        normal[1] = -n_world[1];
        normal[2] = -n_world[2];

        penetration = radius_capsule - best_dist;

        return true;
    }

    template <typename Real>
    __device__ bool collide_capsule_capsule(
        Real* contact_a, Real* contact_b, Real* normal, Real& penetration,
        const Real* pos_a, const Real* orient_a, Real radius_a, Real half_height_a,
        const Real* pos_b, const Real* orient_b, Real radius_b, Real half_height_b)
    {
        Real axis_a[3] = {0, 1, 0};
        Real rotated_axis_a[3];
        cudaPhysics::quatRotateVector(rotated_axis_a, orient_a, axis_a);

        Real top_a[3], bottom_a[3];
        cudaPhysics::axpby(top_a, static_cast<Real>(1.0), pos_a, half_height_a, rotated_axis_a, 3);
        cudaPhysics::axpby(bottom_a, static_cast<Real>(1.0), pos_a, -half_height_a, rotated_axis_a, 3);

        Real axis_b[3] = {0, 1, 0};
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
        if (denom < EPSILON)
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

        if (dist >= sum_radius || dist < EPSILON)
            return false;

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

        get_local_point(contact_a, pos_a, orient_a_conj, contact_on_a);
        get_local_point(contact_b, pos_b, orient_b_conj, contact_on_b);

        normal[0] = n[0];
        normal[1] = n[1];
        normal[2] = n[2];

        penetration = sum_radius - dist;

        return true;
    }

    template <typename Real>
    __device__ bool detect_collision(
        Real* contact_a, Real* contact_b, Real* normal, Real& penetration,
        int shape_a, const Real* pos_a, const Real* orient_a, const Real* params_a,
        int shape_b, const Real* pos_b, const Real* orient_b, const Real* params_b)
    {
        if (shape_a == RIGID_BODY_SPHERE && shape_b == RIGID_BODY_SPHERE)
        {
            return collide_sphere_sphere(contact_a, contact_b, normal, penetration,
                pos_a, orient_a, params_a[0],
                pos_b, orient_b, params_b[0]);
        }
        else if (shape_a == RIGID_BODY_SPHERE && shape_b == RIGID_BODY_BOX)
        {
            bool result = collide_sphere_box(contact_a, contact_b, normal, penetration,
                pos_a, orient_a, params_a[0],
                pos_b, orient_b, params_b[0], params_b[1], params_b[2]);
            if (result)
            {
                normal[0] = -normal[0];
                normal[1] = -normal[1];
                normal[2] = -normal[2];
                Real temp[3];
                cudaPhysics::vecCopy3(temp, contact_a);
                cudaPhysics::vecCopy3(contact_a, contact_b);
                cudaPhysics::vecCopy3(contact_b, temp);
            }
            return result;
        }
        else if (shape_a == RIGID_BODY_BOX && shape_b == RIGID_BODY_SPHERE)
        {
            return collide_sphere_box(contact_b, contact_a, normal, penetration,
                pos_b, orient_b, params_b[0],
                pos_a, orient_a, params_a[0], params_a[1], params_a[2]);
        }
        else if (shape_a == RIGID_BODY_BOX && shape_b == RIGID_BODY_BOX)
        {
            return collide_box_box(contact_a, contact_b, normal, penetration,
                pos_a, orient_a, params_a[0], params_a[1], params_a[2],
                pos_b, orient_b, params_b[0], params_b[1], params_b[2]);
        }
        else if (shape_a == RIGID_BODY_CAPSULE && shape_b == RIGID_BODY_SPHERE)
        {
            bool result = collide_capsule_sphere(contact_a, contact_b, normal, penetration,
                pos_a, orient_a, params_a[0], params_a[1],
                pos_b, orient_b, params_b[0]);
            return result;
        }
        else if (shape_a == RIGID_BODY_SPHERE && shape_b == RIGID_BODY_CAPSULE)
        {
            bool result = collide_capsule_sphere(contact_b, contact_a, normal, penetration,
                pos_b, orient_b, params_b[0], params_b[1],
                pos_a, orient_a, params_a[0]);
            if (result)
            {
                normal[0] = -normal[0];
                normal[1] = -normal[1];
                normal[2] = -normal[2];
                Real temp[3];
                cudaPhysics::vecCopy3(temp, contact_a);
                cudaPhysics::vecCopy3(contact_a, contact_b);
                cudaPhysics::vecCopy3(contact_b, temp);
            }
            return result;
        }
        else if (shape_a == RIGID_BODY_CAPSULE && shape_b == RIGID_BODY_BOX)
        {
            bool result = collide_capsule_box(contact_a, contact_b, normal, penetration,
                pos_a, orient_a, params_a[0], params_a[1],
                pos_b, orient_b, params_b[0], params_b[1], params_b[2]);
            return result;
        }
        else if (shape_a == RIGID_BODY_BOX && shape_b == RIGID_BODY_CAPSULE)
        {
            bool result = collide_capsule_box(contact_b, contact_a, normal, penetration,
                pos_b, orient_b, params_b[0], params_b[1],
                pos_a, orient_a, params_a[0], params_a[1], params_a[2]);
            if (result)
            {
                normal[0] = -normal[0];
                normal[1] = -normal[1];
                normal[2] = -normal[2];
                Real temp[3];
                cudaPhysics::vecCopy3(temp, contact_a);
                cudaPhysics::vecCopy3(contact_a, contact_b);
                cudaPhysics::vecCopy3(contact_b, temp);
            }
            return result;
        }
        else if (shape_a == RIGID_BODY_CAPSULE && shape_b == RIGID_BODY_CAPSULE)
        {
            return collide_capsule_capsule(contact_a, contact_b, normal, penetration,
                pos_a, orient_a, params_a[0], params_a[1],
                pos_b, orient_b, params_b[0], params_b[1]);
        }

        return false;
    }

    template <typename Real>
    __global__ void detect_body_collision_kernel(
        CollisionInfo<Real>* collisions,
        int* collision_count,
        unsigned int max_collisions,
        unsigned int num_bodies,
        const Real* position,
        const Real* orientation,
        const int* shape,
        const Real* shape_param)
    {
        unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int total_pairs = num_bodies * (num_bodies - 1) / 2;

        if (idx >= total_pairs)
            return;

        int i = (int)((2.0 * num_bodies - 1 - sqrt((2.0 * num_bodies - 1.0) * (2.0 * num_bodies - 1.0) - 8.0 * idx)) / 2.0);
        int j = i + 1 + (idx - i * (num_bodies - 1) + i * (i - 1) / 2);

        const Real* pos_a = &position[i * 3];
        const Real* orient_a = &orientation[i * 4];
        int shape_a = shape[i];
        const Real* params_a = &shape_param[i * 3];

        const Real* pos_b = &position[j * 3];
        const Real* orient_b = &orientation[j * 4];
        int shape_b = shape[j];
        const Real* params_b = &shape_param[j * 3];

        Real contact_a[3], contact_b[3], normal[3];
        Real penetration;

        if (!detect_collision(contact_a, contact_b, normal, penetration,
            shape_a, pos_a, orient_a, params_a,
            shape_b, pos_b, orient_b, params_b))
            return;

        int slot = atomicAdd(collision_count, 1);
        if (slot >= static_cast<int>(max_collisions))
            return;

        collisions[slot].body_id_a = static_cast<int>(i);
        collisions[slot].body_id_b = static_cast<int>(j);
        collisions[slot].local_point_a[0] = contact_a[0];
        collisions[slot].local_point_a[1] = contact_a[1];
        collisions[slot].local_point_a[2] = contact_a[2];
        collisions[slot].local_point_b[0] = contact_b[0];
        collisions[slot].local_point_b[1] = contact_b[1];
        collisions[slot].local_point_b[2] = contact_b[2];
        collisions[slot].normal[0] = normal[0];
        collisions[slot].normal[1] = normal[1];
        collisions[slot].normal[2] = normal[2];
    }
}

template <typename Real>
void BodyCollision<Real>::Detect(
    CollisionData<Real>* collision_data,
    unsigned int max_collisions,
    unsigned int num_bodies,
    const Real* dev_position,
    const Real* dev_orientation,
    const int* dev_shape,
    const Real* dev_shape_param)
{
    if (num_bodies < 2 || max_collisions == 0)
        return;
    unsigned int total_pairs = num_bodies * (num_bodies - 1) / 2;

    BodyCollisionKernel::detect_body_collision_kernel<Real><<<CUDA_GRID_SIZE(total_pairs), CUDA_BLOCK_SIZE>>>(
        collision_data->dev_collisions,
        collision_data->dev_collision_count,
        max_collisions,
        num_bodies,
        dev_position,
        dev_orientation,
        dev_shape,
        dev_shape_param);
}

template class BodyCollision<float>;
template class BodyCollision<double>;

#pragma once
#include "algebra.cuh"

namespace cudaPhysics
{
    template <typename Real>
    __host__ __device__ __forceinline__ void triangle_normal(Real *normal, const Real *tri_a_pos, const Real *tri_b_pos, const Real *tri_c_pos)
    {
        Real ab[3], ac[3];
        vecSubs3(ab, tri_b_pos, tri_a_pos);
        vecSubs3(ac, tri_c_pos, tri_a_pos);
        cross3(normal, ab, ac);
        norm3InPlace(normal);
    }

    template <typename Real>
    __host__ __device__ __forceinline__ Real point_to_triangle_sign_distance(const Real *point_pos, const Real *tri_a_pos, const Real *tri_b_pos, const Real *tri_c_pos)
    {
        Real normal[3], ap[3];
        triangle_normal(normal, tri_a_pos, tri_b_pos, tri_c_pos);   
        vecSubs3(ap, point_pos, tri_a_pos);
        return dot3(normal, ap);
    }

    template <typename Real>
    __host__ __device__ __forceinline__ Real point_to_triangle_unsign_distance(const Real *point_pos, const Real *tri_a_pos, const Real *tri_b_pos, const Real *tri_c_pos)
    {
        return abs(point_to_triangle_sign_distance(point_pos, tri_a_pos, tri_b_pos, tri_c_pos));
    }

    template <typename Real>
    __host__ __device__ __forceinline__ bool is_point_in_triangle(const Real *point_pos, const Real *tri_a_pos, const Real *tri_b_pos, const Real *tri_c_pos)
    {
        Real normal[3], ab[3], bc[3], ca[3], ap[3], bp[3], c_abp[3], c_bcp[3], c_cap[3];
        vecSubs3(ab, tri_b_pos, tri_a_pos);
        vecSubs3(bc, tri_c_pos, tri_b_pos);
        cross3(normal, ab, bc);
        norm3InPlace(normal);
        vecSubs3(ca, tri_a_pos, tri_c_pos);
        vecSubs3(ap, point_pos, tri_a_pos);
        vecSubs3(bp, point_pos, tri_b_pos);
        cross3(c_abp, ab, ap);
        cross3(c_bcp, bc, bp);
        cross3(c_cap, ca, ap);
        return dot3(normal, c_abp) >= 0 && dot3(normal, c_bcp) >= 0 && dot3(normal, c_cap) >= 0;
    }

    template <typename Real>
    __host__ __device__ __forceinline__ void projection_barycentric_coordinates(Real *barycentric, const Real *point_pos, const Real *tri_a_pos, const Real *tri_b_pos, const Real *tri_c_pos)
    {
        Real ab[3], ac[3], aq[3], ap[3], normal[3], tmp[3];
        vecSubs3(ab, tri_b_pos, tri_a_pos);
        vecSubs3(ac, tri_c_pos, tri_a_pos);
        cross3(normal, ab, ac);
        norm3InPlace(normal);
        vecSubs3(aq, point_pos, tri_a_pos);
        vecMul3(tmp, dot3(normal, aq), normal);
        vecSubs3(ap, aq, tmp);
        
        Real d00 = dot3(ab, ab);
        Real d01 = dot3(ab, ac);
        Real d11 = dot3(ac, ac);
        Real d20 = dot3(ap, ab);
        Real d21 = dot3(ap, ac);
        Real denom = d00 * d11 - d01 * d01;
        barycentric[1] = (d11 * d20 - d01 * d21) / denom;
        barycentric[2] = (d00 * d21 - d01 * d20) / denom;
        barycentric[0] = 1 - barycentric[1] - barycentric[2];
    }

    template <typename Real>
    __host__ __device__ __forceinline__ Real point_to_segment_distance(Real *closest_on_segment, const Real *point, const Real *seg_start, const Real *seg_end)
    {
        Real seg[3];
        vecSubs3(seg, seg_end, seg_start);
        Real seg_len_sq = dot3(seg, seg);

        if (seg_len_sq < static_cast<Real>(1e-12))
        {
            vecCopy3(closest_on_segment, seg_start);
            Real diff[3];
            vecSubs3(diff, point, seg_start);
            return len3(diff);
        }

        Real t = dot3(seg, point) - dot3(seg, seg_start);
        t /= seg_len_sq;
        t = max(static_cast<Real>(0.0), min(static_cast<Real>(1.0), t));
        axpby(closest_on_segment, static_cast<Real>(1.0), seg_start, t, seg, 3);

        Real diff[3];
        vecSubs3(diff, point, closest_on_segment);
        return len3(diff);
    }

    template <typename Real>
    __host__ __device__ __forceinline__ Real segment_to_segment_distance(Real *closest_on_seg1, Real *closest_on_seg2,
                                                                          const Real *seg1_start, const Real *seg1_end,
                                                                          const Real *seg2_start, const Real *seg2_end)
    {
        Real d1[3], d2[3], r[3];
        vecSubs3(d1, seg1_end, seg1_start);
        vecSubs3(d2, seg2_end, seg2_start);
        vecSubs3(r, seg1_start, seg2_start);

        Real a = dot3(d1, d1);
        Real b = dot3(d1, d2);
        Real c = dot3(d2, d2);
        Real d = dot3(d1, r);
        Real e = dot3(d2, r);
        Real denom = a * c - b * b;

        if (denom < static_cast<Real>(1e-12))
        {
            Real dist = point_to_segment_distance(closest_on_seg2, seg1_start, seg2_start, seg2_end);
            vecCopy3(closest_on_seg1, seg1_start);

            Real tmp_closest[3];
            Real tmp_dist = point_to_segment_distance(tmp_closest, seg1_end, seg2_start, seg2_end);
            if (tmp_dist < dist)
            {
                dist = tmp_dist;
                vecCopy3(closest_on_seg1, seg1_end);
                vecCopy3(closest_on_seg2, tmp_closest);
            }

            tmp_dist = point_to_segment_distance(tmp_closest, seg2_start, seg1_start, seg1_end);
            if (tmp_dist < dist)
            {
                dist = tmp_dist;
                vecCopy3(closest_on_seg2, seg2_start);
                vecCopy3(closest_on_seg1, tmp_closest);
            }

            tmp_dist = point_to_segment_distance(tmp_closest, seg2_end, seg1_start, seg1_end);
            if (tmp_dist < dist)
            {
                dist = tmp_dist;
                vecCopy3(closest_on_seg2, seg2_end);
                vecCopy3(closest_on_seg1, tmp_closest);
            }

            return dist;
        }

        Real s = (b * e - c * d) / denom;
        Real t = (a * e - b * d) / denom;

        bool s_clamped = false;
        bool t_clamped = false;

        if (s < static_cast<Real>(0.0))
        {
            s = static_cast<Real>(0.0);
            s_clamped = true;
        }
        else if (s > static_cast<Real>(1.0))
        {
            s = static_cast<Real>(1.0);
            s_clamped = true;
        }

        if (t < static_cast<Real>(0.0))
        {
            t = static_cast<Real>(0.0);
            t_clamped = true;
        }
        else if (t > static_cast<Real>(1.0))
        {
            t = static_cast<Real>(1.0);
            t_clamped = true;
        }

        if (s_clamped || t_clamped)
        {
            axpby(closest_on_seg1, static_cast<Real>(1.0), seg1_start, s, d1, 3);
            axpby(closest_on_seg2, static_cast<Real>(1.0), seg2_start, t, d2, 3);

            Real dist_sq = static_cast<Real>(0.0);
            for (int i = 0; i < 3; ++i)
            {
                Real diff_i = closest_on_seg1[i] - closest_on_seg2[i];
                dist_sq += diff_i * diff_i;
            }

            if (s_clamped)
            {
                Real tmp_closest[3];
                Real tmp_dist = point_to_segment_distance(tmp_closest, closest_on_seg1, seg2_start, seg2_end);
                if (tmp_dist * tmp_dist < dist_sq)
                {
                    dist_sq = tmp_dist * tmp_dist;
                    vecCopy3(closest_on_seg2, tmp_closest);
                }
            }

            if (t_clamped)
            {
                Real tmp_closest[3];
                Real tmp_dist = point_to_segment_distance(tmp_closest, closest_on_seg2, seg1_start, seg1_end);
                if (tmp_dist * tmp_dist < dist_sq)
                {
                    dist_sq = tmp_dist * tmp_dist;
                    vecCopy3(closest_on_seg1, tmp_closest);
                }
            }

            return sqrt(dist_sq);
        }

        axpby(closest_on_seg1, static_cast<Real>(1.0), seg1_start, s, d1, 3);
        axpby(closest_on_seg2, static_cast<Real>(1.0), seg2_start, t, d2, 3);

        Real diff[3];
        vecSubs3(diff, closest_on_seg2, closest_on_seg1);
        return len3(diff);
    }
}

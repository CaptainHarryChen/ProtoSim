#pragma once
#include "algebra.cuh"

namespace cudaPhysics
{
    template <typename Real>
    __host__ __device__ __forceinline__ void triangle_normal(const Real *tri_a_pos, const Real *tri_b_pos, const Real *tri_c_pos, Real *normal)
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
        triangle_normal(tri_a_pos, tri_b_pos, tri_c_pos, normal);   
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
        triangle_normal(tri_a_pos, tri_b_pos, tri_c_pos, normal);
        vecSubs3(ab, tri_b_pos, tri_a_pos);
        vecSubs3(bc, tri_c_pos, tri_b_pos);
        vecSubs3(ca, tri_a_pos, tri_c_pos);
        vecSubs3(ap, point_pos, tri_a_pos);
        vecSubs3(bp, point_pos, tri_b_pos);
        cross3(c_abp, ab, ap);
        cross3(c_bcp, bc, bp);
        cross3(c_cap, ca, ap);
        return dot3(normal, c_abp) >= 0 && dot3(normal, c_bcp) >= 0 && dot3(normal, c_cap) >= 0;
    }
}

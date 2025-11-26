#pragma once

namespace cudaPhysics
{
    /**
     * @brief A simple corotated linear constitutive model used in 
     * "A Chebyshev Semi-Iterative Approach for Accelerating Projective and Position-Based Dynamics" (Wang, 2015)
     * P = mu * (F - R)
     */
    template <typename Real>
	__host__ __device__ void calc_corotated_linear_P(Real *P, const Real *F, Real mu, Real lambda)
	{
        Real R[9];
		polar_decomposition_R(R, F);
        axpby(P, 2 * mu, F, -2 * mu, R, 9);
	}

    /**
     * @brief Calculate the K = (partial f / partial x) of the Corotated Linear force
     * (partial P / partial F) = 2 * mu * I
     * So each direction (xyz) has the same stiffness. The K only needs to store 4 values for the 4 vertices.
     * @param K The diagonal part of the output stiffness matrix (length 4)
     * each entry correspond to one vertex of the tetrahedron
     */
    template <typename Real>
    __host__ __device__ void calc_corotated_linear_force_diff(
        Real *K,
        const Real *InvDm,
        Real W,
        Real mu,
        Real lambda)
    {
        Real delta_f[9]; // delta_f = -2 * V0 * stiffness * invDm * invDm^T * (delta Ds^T)
        matmatTMul3(delta_f, InvDm, InvDm);
        vecMul(delta_f, -2 * W * mu, delta_f, 9);
        // stiffness matrix K are 12x12, which has 12 diagonal elements. But every 3 elements are the same. So we only need 4 elements.
        K[1] = delta_f[0]; //(delta Ds^T)[0][0:3] = 1
        K[2] = delta_f[4]; //(delta Ds^T)[1][0:3] = 1
        K[3] = delta_f[8]; //(delta Ds^T)[2][0:3] = 1
        // (delta Ds^T)[:,:] = -1 and f[0] = - f[1] - f[2] - f[3] which means delta_f[0] need a sum
        K[0] = delta_f[0] + delta_f[1] + delta_f[2] + delta_f[3] + delta_f[4] + delta_f[5] + delta_f[6] + delta_f[7] + delta_f[8];
    }
}

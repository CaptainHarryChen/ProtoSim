#pragma once
#include <Math/algebra.cuh>

namespace cudaPhysics
{
    template <typename Real>
    __host__ __device__ __forceinline__ void calc_neohookean_P(Real *P, Real *F, Real mu, Real lambda)
    {
        Real invF[9], invFT[9];
        matInv3(invF, F);
        matTrans3(invFT, invF);
        Real J = det3(F);

        Real temp1[9], temp2[9];
        vecSubs(temp1, F, invFT, 9);
        matMul3(temp1, mu, temp1);
        matMul3(temp2, lambda * log(J), invFT);

        vecAdd(P, temp1, temp2, 9);
    }

    template <typename Real>
    __host__ __device__ void get_rotation_matrix_from_deformation_gradient(Real *R, const Real *F)
    {
        Real C[9];
        memset(C, 0, sizeof(Real) * 9);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    C[3 * i + j] += F[3 * k + i] * F[3 * k + j];

        Real C2[9];
        memset(C2, 0, sizeof(Real) * 9);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    C2[3 * i + j] += C[3 * i + k] * C[3 * j + k];

        Real det = F[0] * F[4] * F[8] +
                   F[1] * F[5] * F[6] +
                   F[3] * F[7] * F[2] -
                   F[2] * F[4] * F[6] -
                   F[1] * F[3] * F[8] -
                   F[0] * F[5] * F[7];

        Real I_c = C[0] + C[4] + C[8];
        Real I_c2 = I_c * I_c;
        Real II_c = 0.5 * (I_c2 - C2[0] - C2[4] - C2[8]);
        Real III_c = det * det;
        Real k = I_c2 - 3 * II_c;

        Real inv_U[9];
        if (k < 1e-12)
        {
            Real inv_lambda = 1 / sqrt(I_c / 3);
            memset(inv_U, 0, sizeof(Real) * 9);
            inv_U[0] = inv_lambda;
            inv_U[4] = inv_lambda;
            inv_U[8] = inv_lambda;
        }
        else
        {
            Real l = I_c * (I_c * I_c - 4.5 * II_c) + 13.5 * III_c;
            Real k_root = sqrt(k);
            Real value = l / (k * k_root);
            if (value < -1.0)
                value = -1.0;
            if (value > 1.0)
                value = 1.0;
            Real phi = acos(value);
            Real lambda2 = (I_c + 2 * k_root * cos(phi / 3)) / 3.0;
            Real lambda = sqrt(lambda2);

            Real III_u = sqrt(III_c);
            if (det < 0)
                III_u = -III_u;
            Real I_u = lambda + sqrt(-lambda2 + I_c + 2 * III_u / lambda);
            Real II_u = (I_u * I_u - I_c) * 0.5;

            Real U[9];
            Real inv_rate, factor;

            inv_rate = 1 / (I_u * II_u - III_u);
            factor = I_u * III_u * inv_rate;

            memset(U, 0, sizeof(Real) * 9);
            U[0] = factor;
            U[4] = factor;
            U[8] = factor;

            factor = (I_u * I_u - II_u) * inv_rate;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    U[3 * i + j] += factor * C[3 * i + j] - inv_rate * C2[3 * i + j];

            inv_rate = 1 / III_u;
            factor = II_u * inv_rate;
            memset(inv_U, 0, sizeof(Real) * 9);
            inv_U[0] = factor;
            inv_U[4] = factor;
            inv_U[8] = factor;

            factor = -I_u * inv_rate;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    inv_U[3 * i + j] += factor * U[3 * i + j] + inv_rate * C[3 * i + j];
        }

        memset(R, 0, sizeof(Real) * 9);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    R[3 * i + j] += F[3 * i + k] * inv_U[3 * k + j];
    }
}

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
}

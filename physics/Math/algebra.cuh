#pragma once

namespace cudaPhysics
{
    template <class T>
    __host__ __device__ __forceinline__ void vecCopy(T *X, const T *A, int n)
    {
        for (int i = 0; i < n; ++i)
            X[i] = A[i];
    }

    template <class T>
    __host__ __device__ __forceinline__ void vecCopy3(T *X, const T *A)
    {
        X[0] = A[0];
        X[1] = A[1];
        X[2] = A[2];
    }

    template <class T>
    __host__ __device__ __forceinline__ void vecAdd(T *X, const T *A, const T *B, int n)
    {
        for (int i = 0; i < n; ++i)
            X[i] = A[i] + B[i];
    }

    template <class T>
    __host__ __device__ __forceinline__ void vecAdd3(T *X, const T *A, const T *B)
    {
        X[0] = A[0] + B[0];
        X[1] = A[1] + B[1];
        X[2] = A[2] + B[2];
    }

    template <class T>
    __host__ __device__ __forceinline__ void vecSubs(T *X, const T *A, const T *B, int n)
    {
        for (int i = 0; i < n; ++i)
            X[i] = A[i] - B[i];
    }

    template <class T>
    __host__ __device__ __forceinline__ void vecSubs3(T *X, const T *A, const T *B)
    {
        X[0] = A[0] - B[0];
        X[1] = A[1] - B[1];
        X[2] = A[2] - B[2];
    }

    template <class T>
    __host__ __device__ __forceinline__ void vecMul(T *X, const T a, const T *B, int n)
    {
        for (int i = 0; i < n; ++i)
            X[i] = a * B[i];
    }

    template <class T>
    __host__ __device__ __forceinline__ void vecMul3(T *X, const T *A, const T *B)
    {
        X[0] = A[0] * B[0];
        X[1] = A[1] * B[1];
        X[2] = A[2] * B[2];
    }

    template <class T>
    __host__ __device__ __forceinline__ void vecMul3(T *X, const T A, const T *B)
    {
        X[0] = A * B[0];
        X[1] = A * B[1];
        X[2] = A * B[2];
    }

    template <class T>
    __host__ __device__ __forceinline__ void vecMul3(T *X, const T A)
    {
        X[0] = A * X[0];
        X[1] = A * X[1];
        X[2] = A * X[2];
    }
    template <class T>
    __host__ __device__ __forceinline__ void vecvecT(T *X, const T *v, const int n)
    {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                X[n * i + j] = v[i] * v[j];
    }

    template <class T>
    __host__ __device__ __forceinline__ void vecvecT(T *X, const T *v0, const T *v1, const int m, const int n)
    {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                X[n * i + j] = v0[i] * v1[j];
    }

    template <class T>
    __host__ __device__ __forceinline__ void cross3(T *result, const T *vec1, const T *vec2)
    {
        result[0] = vec1[1] * vec2[2] - vec2[1] * vec1[2];
        result[1] = -vec1[0] * vec2[2] + vec2[0] * vec1[2];
        result[2] = vec1[0] * vec2[1] - vec2[0] * vec1[1];
    }

    template <class T>
    __host__ __device__ __forceinline__ T dot3(const T *vec1, const T *vec2)
    {
        return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
    }

    template <class T>
    __host__ __device__ __forceinline__ void norm3(T *result, const T *vec)
    {
        T one_over_length = 1.0 / sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
        result[0] = vec[0] * one_over_length;
        result[1] = vec[1] * one_over_length;
        result[2] = vec[2] * one_over_length;
    }

    template <class T>
    __host__ __device__ __forceinline__ T len3(const T *vec)
    {
        return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
    }

    template <class T>
    __host__ __device__ __forceinline__ T dist3(const T *vec1, const T *vec2)
    {
        T vec[3];
        vecSubs3<T>(vec, vec1, vec2);
        return len3<T>(vec);
    }

    template <class T>
    __host__ __device__ __forceinline__ T normSquared3(const T *vec)
    {
        return vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
    }

    template <class T>
    __host__ __device__ __forceinline__ void normSquared(T &result, const T *vec, const unsigned int N)
    {
        result = 0.0;
        for (int i = 0; i < N; ++i)
            result += vec[i] * vec[i];
    }

    template <class T>
    __host__ __device__ __forceinline__ void norm3InPlace(T *vec)
    {
        T one_over_length = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
        if (one_over_length == 0.)
            return;
        one_over_length = 1.0 / sqrt(one_over_length);
        vec[0] = vec[0] * one_over_length;
        vec[1] = vec[1] * one_over_length;
        vec[2] = vec[2] * one_over_length;
    }

    template <class T>
    __host__ __device__ __forceinline__ void matVec3(T *result, T *mat, T *vec)
    {
        result[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
        result[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
        result[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
    }

    template <class T>
    __host__ __device__ __forceinline__ void matInv3(T *inv_X, const T *X)
    {
        inv_X[0] = X[4] * X[8] - X[5] * X[7];
        inv_X[1] = -X[1] * X[8] + X[2] * X[7];
        inv_X[2] = X[1] * X[5] - X[2] * X[4];
        inv_X[3] = -X[3] * X[8] + X[5] * X[6];
        inv_X[4] = X[0] * X[8] - X[2] * X[6];
        inv_X[5] = -X[0] * X[5] + X[2] * X[3];
        inv_X[6] = X[3] * X[7] - X[4] * X[6];
        inv_X[7] = -X[0] * X[7] + X[1] * X[6];
        inv_X[8] = X[0] * X[4] - X[1] * X[3];
        T J = X[0] * inv_X[0] + X[3] * inv_X[1] + X[6] * inv_X[2];
        T inv_J = 1.0 / J;
        for (int i = 0; i < 9; i++)
            inv_X[i] *= inv_J;
    }

    /**** MatLab codes ****
     * syms x0 x1 x2 x3 x4 x5 x6 x7 x8
     * syms X
     * X = [x0 x1 x2; x3 x4 x5; x6 x7 x8]
     * J = det(X)
     * ccode(inv(X) * J)
     **** MatLab codes ****/

    template <class T>
    __host__ __device__ __forceinline__ void matInv2(T *inv_X, const T *X)
    {
        T J = X[0] * X[3] - X[1] * X[2];
        T inv_J = 1. / J;
        inv_X[0] = X[3] * inv_J;
        inv_X[1] = -X[1] * inv_J;
        inv_X[2] = -X[2] * inv_J;
        inv_X[3] = X[0] * inv_J;
    }

    template <class T>
    __host__ __device__ __forceinline__ void matMul3(T *X, const T *A, const T *B)
    {
        X[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
        X[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
        X[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];
        X[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
        X[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
        X[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];
        X[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
        X[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
        X[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
    }

    template <class T>
    __host__ __device__ __forceinline__ void matMul3(T *X, const T a, const T *B)
    {
        for (int i = 0; i < 9; ++i)
            X[i] = a * B[i];
    }

    template <class T>
    __host__ __device__ __forceinline__ void matSubs3(T *X, const T *A, const T *B)
    {
        for (int i = 0; i < 9; ++i)
            X[i] = A[i] - B[i];
    }

    // X = (AB)^T
    template <class T>
    __host__ __device__ __forceinline__ void matMul3T(T *X, const T *A, const T *B)
    {
        X[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
        X[3] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
        X[6] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];
        X[1] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
        X[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
        X[7] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];
        X[2] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
        X[5] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
        X[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
    }

    /**** MatLab codes ****
     * syms a0 a1 a2 a3 a4 a5 a6 a7 a8
     * syms A
     * A = [a0 a1 a2; a3 a4 a5; a6 a7 a8]
     * syms b0 b1 b2 b3 b4 b5 b6 b7 b8
     * syms B
     * B = [b0 b1 b2; b3 b4 b5; b6 b7 b8]
     * X = A * B
     * ccode(X)
     **** MatLab codes ****/

    template <class T>
    __host__ __device__ __forceinline__ void matTMul3(T *X, const T *A, const T *B)
    {
        X[0] = A[0] * B[0] + A[3] * B[3] + A[6] * B[6];
        X[1] = A[0] * B[1] + A[3] * B[4] + A[6] * B[7];
        X[2] = A[0] * B[2] + A[3] * B[5] + A[6] * B[8];
        X[3] = A[1] * B[0] + A[4] * B[3] + A[7] * B[6];
        X[4] = A[1] * B[1] + A[4] * B[4] + A[7] * B[7];
        X[5] = A[1] * B[2] + A[4] * B[5] + A[7] * B[8];
        X[6] = A[2] * B[0] + A[5] * B[3] + A[8] * B[6];
        X[7] = A[2] * B[1] + A[5] * B[4] + A[8] * B[7];
        X[8] = A[2] * B[2] + A[5] * B[5] + A[8] * B[8];
    }

    template <class T>
    __host__ __device__ __forceinline__ void matMulT3(T *X, const T *A, const T *B)
    {
        X[0] = A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
        X[1] = A[0] * B[3] + A[1] * B[4] + A[2] * B[5];
        X[2] = A[0] * B[6] + A[1] * B[7] + A[2] * B[8];
        X[3] = A[3] * B[0] + A[4] * B[1] + A[5] * B[2];
        X[4] = A[3] * B[3] + A[4] * B[4] + A[5] * B[5];
        X[5] = A[3] * B[6] + A[4] * B[7] + A[5] * B[8];
        X[6] = A[6] * B[0] + A[7] * B[1] + A[8] * B[2];
        X[7] = A[6] * B[3] + A[7] * B[4] + A[8] * B[5];
        X[8] = A[6] * B[6] + A[7] * B[7] + A[8] * B[8];
    }

    template <class T>
    __host__ __device__ __forceinline__ void swap(T &x, T &y)
    {
        T t = x;
        x = y;
        y = t;
    }

    template <class T>
    __host__ __device__ __forceinline__ void matSquareTransInPlace(T *X, int n) // transpose only square matrix
    {
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
            {
                swap(X[n * i + j], X[n * j + i]);
            }
    }

    template <class T>
    __host__ __device__ __forceinline__ T det2(const T *X)
    {
        return X[0] * X[3] - X[1] * X[2];
    }

    template <class T>
    __host__ __device__ __forceinline__ T det3(const T *X)
    {
        return X[0] * X[4] * X[8] - X[0] * X[5] * X[7] - X[1] * X[3] * X[8] + X[1] * X[5] * X[6] + X[2] * X[3] * X[7] - X[2] * X[4] * X[6];
    }

    template <class T>
    __host__ __device__ __forceinline__ void svd3_cuda(T *U, T *D, T *V, const T *X)
    {
        T _[9];
        svd<T>(X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8],
               U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7], U[8],
               D[0], _[1], _[2], _[3], D[1], _[5], _[6], _[7], D[2],
               V[0], V[1], V[2], V[3], V[4], V[5], V[6], V[7], V[8]);
    }

    template <class T>
    __host__ __device__ __forceinline__ void svd3rv_cuda(T *U, T *D, T *V, const T *X) // rotation invariant svd
    {
        svd3_cuda<T>(U, D, V, X);
        T L[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        T UVT[9];
        matMulT3<T>(UVT, U, V);
        T detUVT;
        det3<T>(detUVT, UVT);
        L[9] = detUVT;
        T detU, detV;
        det3<T>(detU, U);
        det3<T>(detV, V);

        if (detU < 0.0 && detV > 0.0)
        {
            T *_U;
            matMul3<T>(_U, U, L);
            for (int i = 0; i < 9; ++i)
                U[i] = _U[i];
        }
        if (detU > 0.0 && detV < 0.0)
        {
            T *_V;
            matMul3<T>(_V, V, L);
            for (int i = 0; i < 9; ++i)
                V[i] = _V[i];
        }

        D[9] = D[9] * L[9];
    }

    template <class T>
    __host__ __device__ __forceinline__ void matUXVT3(T *R, const T *U, const T *X, const T *V)
    {
        R[0] = V[0] * (U[0] * X[0] + U[1] * X[3] + U[2] * X[6]) + V[1] * (U[0] * X[1] + U[1] * X[4] + U[2] * X[7]) + V[2] * (U[0] * X[2] + U[1] * X[5] + U[2] * X[8]);
        R[1] = V[3] * (U[0] * X[0] + U[1] * X[3] + U[2] * X[6]) + V[4] * (U[0] * X[1] + U[1] * X[4] + U[2] * X[7]) + V[5] * (U[0] * X[2] + U[1] * X[5] + U[2] * X[8]);
        R[2] = V[6] * (U[0] * X[0] + U[1] * X[3] + U[2] * X[6]) + V[7] * (U[0] * X[1] + U[1] * X[4] + U[2] * X[7]) + V[8] * (U[0] * X[2] + U[1] * X[5] + U[2] * X[8]);
        R[3] = V[0] * (U[3] * X[0] + U[4] * X[3] + U[5] * X[6]) + V[1] * (U[3] * X[1] + U[4] * X[4] + U[5] * X[7]) + V[2] * (U[3] * X[2] + U[4] * X[5] + U[5] * X[8]);
        R[4] = V[3] * (U[3] * X[0] + U[4] * X[3] + U[5] * X[6]) + V[4] * (U[3] * X[1] + U[4] * X[4] + U[5] * X[7]) + V[5] * (U[3] * X[2] + U[4] * X[5] + U[5] * X[8]);
        R[5] = V[6] * (U[3] * X[0] + U[4] * X[3] + U[5] * X[6]) + V[7] * (U[3] * X[1] + U[4] * X[4] + U[5] * X[7]) + V[8] * (U[3] * X[2] + U[4] * X[5] + U[5] * X[8]);
        R[6] = V[0] * (U[6] * X[0] + U[7] * X[3] + U[8] * X[6]) + V[1] * (U[6] * X[1] + U[7] * X[4] + U[8] * X[7]) + V[2] * (U[6] * X[2] + U[7] * X[5] + U[8] * X[8]);
        R[7] = V[3] * (U[6] * X[0] + U[7] * X[3] + U[8] * X[6]) + V[4] * (U[6] * X[1] + U[7] * X[4] + U[8] * X[7]) + V[5] * (U[6] * X[2] + U[7] * X[5] + U[8] * X[8]);
        R[8] = V[6] * (U[6] * X[0] + U[7] * X[3] + U[8] * X[6]) + V[7] * (U[6] * X[1] + U[7] * X[4] + U[8] * X[7]) + V[8] * (U[6] * X[2] + U[7] * X[5] + U[8] * X[8]);
    }

    template <class T>
    __host__ __device__ __forceinline__ void matUTXV3(T *R, const T *U, const T *X, const T *V)
    {
        R[0] = V[0] * (U[0] * X[0] + U[3] * X[3] + U[6] * X[6]) + V[3] * (U[0] * X[1] + U[3] * X[4] + U[6] * X[7]) + V[6] * (U[0] * X[2] + U[3] * X[5] + U[6] * X[8]);
        R[1] = V[1] * (U[0] * X[0] + U[3] * X[3] + U[6] * X[6]) + V[4] * (U[0] * X[1] + U[3] * X[4] + U[6] * X[7]) + V[7] * (U[0] * X[2] + U[3] * X[5] + U[6] * X[8]);
        R[2] = V[2] * (U[0] * X[0] + U[3] * X[3] + U[6] * X[6]) + V[5] * (U[0] * X[1] + U[3] * X[4] + U[6] * X[7]) + V[8] * (U[0] * X[2] + U[3] * X[5] + U[6] * X[8]);
        R[3] = V[0] * (U[1] * X[0] + U[4] * X[3] + U[7] * X[6]) + V[3] * (U[1] * X[1] + U[4] * X[4] + U[7] * X[7]) + V[6] * (U[1] * X[2] + U[4] * X[5] + U[7] * X[8]);
        R[4] = V[1] * (U[1] * X[0] + U[4] * X[3] + U[7] * X[6]) + V[4] * (U[1] * X[1] + U[4] * X[4] + U[7] * X[7]) + V[7] * (U[1] * X[2] + U[4] * X[5] + U[7] * X[8]);
        R[5] = V[2] * (U[1] * X[0] + U[4] * X[3] + U[7] * X[6]) + V[5] * (U[1] * X[1] + U[4] * X[4] + U[7] * X[7]) + V[8] * (U[1] * X[2] + U[4] * X[5] + U[7] * X[8]);
        R[6] = V[0] * (U[2] * X[0] + U[5] * X[3] + U[8] * X[6]) + V[3] * (U[2] * X[1] + U[5] * X[4] + U[8] * X[7]) + V[6] * (U[2] * X[2] + U[5] * X[5] + U[8] * X[8]);
        R[7] = V[1] * (U[2] * X[0] + U[5] * X[3] + U[8] * X[6]) + V[4] * (U[2] * X[1] + U[5] * X[4] + U[8] * X[7]) + V[7] * (U[2] * X[2] + U[5] * X[5] + U[8] * X[8]);
        R[8] = V[2] * (U[2] * X[0] + U[5] * X[3] + U[8] * X[6]) + V[5] * (U[2] * X[1] + U[5] * X[4] + U[8] * X[7]) + V[8] * (U[2] * X[2] + U[5] * X[5] + U[8] * X[8]);
    }

    template <class T>
    __host__ __device__ __forceinline__ void matUDVT3(T *R, const T *U, const T *D, const T *V)
    {
        R[0] = D[0] * U[0] * V[0] + D[1] * U[1] * V[1] + D[2] * U[2] * V[2];
        R[1] = D[0] * U[0] * V[3] + D[1] * U[1] * V[4] + D[2] * U[2] * V[5];
        R[2] = D[0] * U[0] * V[6] + D[1] * U[1] * V[7] + D[2] * U[2] * V[8];
        R[3] = D[0] * U[3] * V[0] + D[1] * U[4] * V[1] + D[2] * U[5] * V[2];
        R[4] = D[0] * U[3] * V[3] + D[1] * U[4] * V[4] + D[2] * U[5] * V[5];
        R[5] = D[0] * U[3] * V[6] + D[1] * U[4] * V[7] + D[2] * U[5] * V[8];
        R[6] = D[0] * U[6] * V[0] + D[1] * U[7] * V[1] + D[2] * U[8] * V[2];
        R[7] = D[0] * U[6] * V[3] + D[1] * U[7] * V[4] + D[2] * U[8] * V[5];
        R[8] = D[0] * U[6] * V[6] + D[1] * U[7] * V[7] + D[2] * U[8] * V[8];
    }

    template <class T>
    __host__ __device__ __forceinline__ void matUTDV3(T *R, const T *U, const T *D, const T *V)
    {
        R[0] = D[0] * U[0] * V[0] + D[1] * U[3] * V[3] + D[2] * U[6] * V[6];
        R[1] = D[0] * U[0] * V[1] + D[1] * U[3] * V[4] + D[2] * U[6] * V[7];
        R[2] = D[0] * U[0] * V[2] + D[1] * U[3] * V[5] + D[2] * U[6] * V[8];
        R[3] = D[0] * U[1] * V[0] + D[1] * U[4] * V[3] + D[2] * U[7] * V[6];
        R[4] = D[0] * U[1] * V[1] + D[1] * U[4] * V[4] + D[2] * U[7] * V[7];
        R[5] = D[0] * U[1] * V[2] + D[1] * U[4] * V[5] + D[2] * U[7] * V[8];
        R[6] = D[0] * U[2] * V[0] + D[1] * U[5] * V[3] + D[2] * U[8] * V[6];
        R[7] = D[0] * U[2] * V[1] + D[1] * U[5] * V[4] + D[2] * U[8] * V[7];
        R[8] = D[0] * U[2] * V[2] + D[1] * U[5] * V[5] + D[2] * U[8] * V[8];
    }

    template <class T>
    __host__ __device__ __forceinline__ void matTrans3(T *X_trans, const T *X)
    {
        X_trans[0] = X[0];
        X_trans[3] = X[1];
        X_trans[6] = X[2];
        X_trans[1] = X[3];
        X_trans[4] = X[4];
        X_trans[7] = X[5];
        X_trans[2] = X[6];
        X_trans[5] = X[7];
        X_trans[8] = X[8];
    }

    template <typename UInt>
    __device__ inline int common_upper_bits(const UInt x, const UInt y)
    {
        if (sizeof(UInt) == sizeof(unsigned int))
            return __clz(x ^ y);
        else
            return __clzll(x ^ y);
    }

    template <class T>
    __host__ __device__ __forceinline__ void matVecMul(T *X, const T *A, const T *b, const int rows, const int cols)
    {
        // memset(X, 0, sizeof(T) * rows * cols);
        for (int i = 0; i < rows; ++i)
            X[i] = 0.0;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                X[i] += A[i * cols + j] * b[j];
    }

    template <class T>
    __host__ __device__ __forceinline__ void matTVecMul(T *X, const T *A, const T *b, const int rows, const int cols) // transposed rows and cols
    {
        // memset(X, 0, sizeof(T) * rows * cols);
        for (int i = 0; i < rows; ++i)
            X[i] = 0.0;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                X[i] += A[i + j * rows] * b[j];
    }

    template <class T>
    __host__ __device__ __forceinline__ void matMul(T *X, const T *A, const T *B, const int rows, const int inner, int cols)
    {
        // memset(X, 0, sizeof(T) * rows * cols);
        for (int i = 0; i < rows * cols; ++i)
            X[i] = 0.0;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                for (int k = 0; k < inner; ++k)
                    X[i * cols + j] += A[i * inner + k] * B[k * cols + j];
    }

    template <class T>
    __host__ __device__ __forceinline__ void matTMul(T *X, const T *A, const T *B, const int rows, const int inner, int cols) // transposed rows, inner and cols
    {
        // memset(X, 0, sizeof(T) * rows * cols);
        for (int i = 0; i < rows * cols; ++i)
            X[i] = 0.0;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                for (int k = 0; k < inner; ++k)
                    X[i * cols + j] += A[k * rows + i] * B[k * cols + j];
    }

    template <class T>
    __host__ __device__ __forceinline__ void matMulT(T *X, const T *A, const T *B, const int rows, const int inner, const int cols)
    {
        // memset(X, 0, sizeof(T) * rows * cols);
        for (int i = 0; i < rows * cols; ++i)
            X[i] = 0.0;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                for (int k = 0; k < inner; ++k)
                    X[i * cols + j] += A[i * inner + k] * B[j * inner + k];
    }

    template <typename Real>
    __host__ __device__ __forceinline__ void aUXVT3(Real *result, const Real a, const Real *U, const Real *X, const Real *V)
    {
        result[0] = V[0] * (X[0] * U[0] * a + X[3] * U[1] * a + X[6] * U[2] * a) + V[1] * (X[1] * U[0] * a + X[4] * U[1] * a + X[7] * U[2] * a) + V[2] * (X[2] * U[0] * a + X[5] * U[1] * a + X[8] * U[2] * a);
        result[1] = V[3] * (X[0] * U[0] * a + X[3] * U[1] * a + X[6] * U[2] * a) + V[4] * (X[1] * U[0] * a + X[4] * U[1] * a + X[7] * U[2] * a) + V[5] * (X[2] * U[0] * a + X[5] * U[1] * a + X[8] * U[2] * a);
        result[2] = V[6] * (X[0] * U[0] * a + X[3] * U[1] * a + X[6] * U[2] * a) + V[7] * (X[1] * U[0] * a + X[4] * U[1] * a + X[7] * U[2] * a) + V[8] * (X[2] * U[0] * a + X[5] * U[1] * a + X[8] * U[2] * a);
        result[3] = V[0] * (X[0] * U[3] * a + X[3] * U[4] * a + X[6] * U[5] * a) + V[1] * (X[1] * U[3] * a + X[4] * U[4] * a + X[7] * U[5] * a) + V[2] * (X[2] * U[3] * a + X[5] * U[4] * a + X[8] * U[5] * a);
        result[4] = V[3] * (X[0] * U[3] * a + X[3] * U[4] * a + X[6] * U[5] * a) + V[4] * (X[1] * U[3] * a + X[4] * U[4] * a + X[7] * U[5] * a) + V[5] * (X[2] * U[3] * a + X[5] * U[4] * a + X[8] * U[5] * a);
        result[5] = V[6] * (X[0] * U[3] * a + X[3] * U[4] * a + X[6] * U[5] * a) + V[7] * (X[1] * U[3] * a + X[4] * U[4] * a + X[7] * U[5] * a) + V[8] * (X[2] * U[3] * a + X[5] * U[4] * a + X[8] * U[5] * a);
        result[6] = V[0] * (X[0] * U[6] * a + X[3] * U[7] * a + X[6] * U[8] * a) + V[1] * (X[1] * U[6] * a + X[4] * U[7] * a + X[7] * U[8] * a) + V[2] * (X[2] * U[6] * a + X[5] * U[7] * a + X[8] * U[8] * a);
        result[7] = V[3] * (X[0] * U[6] * a + X[3] * U[7] * a + X[6] * U[8] * a) + V[4] * (X[1] * U[6] * a + X[4] * U[7] * a + X[7] * U[8] * a) + V[5] * (X[2] * U[6] * a + X[5] * U[7] * a + X[8] * U[8] * a);
        result[8] = V[6] * (X[0] * U[6] * a + X[3] * U[7] * a + X[6] * U[8] * a) + V[7] * (X[1] * U[6] * a + X[4] * U[7] * a + X[7] * U[8] * a) + V[8] * (X[2] * U[6] * a + X[5] * U[7] * a + X[8] * U[8] * a);
    }

    template <class T>
    __host__ __device__ __forceinline__ void dyadicN(T *X, T *a, T *b, unsigned int N)
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                X[N * i + j] = a[i] * b[j];
    }

    template <class T>
    __host__ __device__ __forceinline__ void accumulate(T *X, T a, T *B, unsigned int N)
    {
        for (int i = 0; i < N; ++i)
            X[i] += a * B[i];
    }

    template <class T>
    __host__ __device__ __forceinline__ void accumulate(T *X, T *A, T *B, unsigned int N)
    {
        for (int i = 0; i < N; ++i)
            X[i] += A[i] * B[i];
    }

    template <class T>
    __host__ __device__ __forceinline__ void setZero(T *v, unsigned int n)
    {
        for (int i = 0; i < n; ++i)
            v[i] = 0.0;
    }

    template <class T>
    __host__ __device__ __forceinline__ void axpby(T *v, unsigned int n, T a, T *x, T b, T *y)
    {
        for (int i = 0; i < n; ++i)
        {
            v[i] = a * x[i] + b * y[i];
        }
    }

    template <class T>
    __host__ __device__ __forceinline__ void axpbypcz(T *v, unsigned int n, T a, T *x, T b, T *y, T c, T *z)
    {
        for (int i = 0; i < n; ++i)
        {
            v[i] = a * x[i] + b * y[i] + c * z[i];
        }
    }

    // the scalar triple product
    template <class T>
    __host__ __device__ __forceinline__
        T
        stp3(T *u, T *v, T *w)
    {
        // u \dot v \cross w
        return u[2] * (v[0] * w[1] - v[1] * w[0]) - u[1] * (v[0] * w[2] - v[2] * w[0]) + u[0] * (v[1] * w[2] - v[2] * w[1]);
    }

    template <class T>
    __host__ __device__ __forceinline__ void lerp3(T *x, T *x0, T *x1, T t)
    {
        for (unsigned int i = 0; i < 3; ++i)
        {
            x[i] = x0[i] + t * (x1[i] - x0[i]);
        }
    }

    template <typename Real>
    __global__ void axpby_kernel(unsigned int n, Real *r, Real a, Real *x, Real b, Real *y)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= n)
            return;
        r[i] = a * x[i] + b * y[i];
    }

    template <typename Real>
    __global__ void axpbypcz_kernel(unsigned int n, Real *r, Real a, Real *x, Real b, Real *y, Real c, Real *z)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= n)
            return;
        r[i] = a * x[i] + b * y[i] + c * z[i];
    }

    template <typename Real>
    __global__ void negate_kernel(unsigned int n, Real *r, Real *x)
    {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= n)
            return;
        r[i] = -x[i];
    }

    template <class T>
    __host__ __device__ __forceinline__ T normSquared(const T *x, unsigned int N)
    {
        T n = 0.;
        for (unsigned int i = 0; i < N; ++i)
        {
            n += x[i] * x[i];
        }
        return n;
    }

    template <typename Real>
    __host__ __device__ __forceinline__ Real radians(Real degrees)
    {
        return degrees * 3.14159265358979323846 / 180.;
    }

    template <typename Real>
    __host__ __device__ __forceinline__ Real degrees(Real radians)
    {
        return radians / 3.14159265358979323846 * 180.;
    }
}

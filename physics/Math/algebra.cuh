#pragma once

namespace cudaPhysics
{
    template <class T>
    __host__ __device__ __forceinline__ void swap(T &x, T &y)
    {
        T t = x;
        x = y;
        y = t;
    }

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
    __host__ __device__ __forceinline__ void vecvecT(T *X, const T *v0, const T *v1, const int n)
    {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                X[n * i + j] = v0[i] * v1[j];
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
    __host__ __device__ __forceinline__ T dot(const T *vec1, const T *vec2, int n)
    {
        T result = 0.0;
        for (int i = 0; i < n; ++i)
            result += vec1[i] * vec2[i];
        return result;
    }

    template <class T>
    __host__ __device__ __forceinline__ T dot3(const T *vec1, const T *vec2)
    {
        return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
    }

    template <class T>
    __host__ __device__ __forceinline__ T len3(const T *vec)
    {
        return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
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
    __host__ __device__ __forceinline__ T dist3(const T *vec1, const T *vec2)
    {
        T vec[3];
        vecSubs3<T>(vec, vec1, vec2);
        return len3<T>(vec);
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
    __host__ __device__ __forceinline__ void matVecMul(T *X, const T *A, const T *b, const int rows, const int cols)
    {
        for (int i = 0; i < rows; ++i)
            X[i] = 0.0;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                X[i] += A[i * cols + j] * b[j];
    }

    template <class T>
    __host__ __device__ __forceinline__ void matTVecMul(T *X, const T *A, const T *b, const int rows, const int cols) // transposed rows and cols
    {
        for (int i = 0; i < rows; ++i)
            X[i] = 0.0;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                X[i] += A[i + j * rows] * b[j];
    }

    template <class T>
    __host__ __device__ __forceinline__ void matMul(T *X, const T *A, const T *B, const int rows, const int inner, int cols)
    {
        for (int i = 0; i < rows * cols; ++i)
            X[i] = 0.0;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                for (int k = 0; k < inner; ++k)
                    X[i * cols + j] += A[i * inner + k] * B[k * cols + j];
    }

    // transpose the left matrix
    template <class T>
    __host__ __device__ __forceinline__ void matTmatMul(T *X, const T *A, const T *B, const int rows, const int inner, int cols)
    {
        for (int i = 0; i < rows * cols; ++i)
            X[i] = 0.0;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                for (int k = 0; k < inner; ++k)
                    X[i * cols + j] += A[k * rows + i] * B[k * cols + j];
    }

    // transpose the right matrix
    template <class T>
    __host__ __device__ __forceinline__ void matmatTMul(T *X, const T *A, const T *B, const int rows, const int inner, const int cols)
    {
        for (int i = 0; i < rows * cols; ++i)
            X[i] = 0.0;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                for (int k = 0; k < inner; ++k)
                    X[i * cols + j] += A[i * inner + k] * B[j * inner + k];
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
    __host__ __device__ __forceinline__ void matVec3(T *result, const T *mat, const T *vec)
    {
        result[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
        result[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
        result[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
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

    // transpose the left matrix
    template <class T>
    __host__ __device__ __forceinline__ void matTmatMul3(T *X, const T *A, const T *B)
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

    // transpose the right matrix
    template <class T>
    __host__ __device__ __forceinline__ void matmatTMul3(T *X, const T *A, const T *B)
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

    /**** MatLab codes ****
     * syms x0 x1 x2 x3 x4 x5 x6 x7 x8
     * syms X
     * X = [x0 x1 x2; x3 x4 x5; x6 x7 x8]
     * J = det(X)
     * ccode(inv(X) * J)
     **** MatLab codes ****/
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
        T inv_J = 1.0f / J;
        for (int i = 0; i < 9; i++)
            inv_X[i] *= inv_J;
    }

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
    __host__ __device__ __forceinline__ void axpby(T *v, T a, const T *x, T b, const T *y, unsigned int n)
    {
        for (int i = 0; i < n; ++i)
        {
            v[i] = a * x[i] + b * y[i];
        }
    }

    template <class T>
    __host__ __device__ __forceinline__ void axpbypcz(T *v, T a, const T *x, T b, const T *y, T c, const T *z, unsigned int n)
    {
        for (int i = 0; i < n; ++i)
        {
            v[i] = a * x[i] + b * y[i] + c * z[i];
        }
    }

    template <typename Real>
    __host__ __device__ void polar_decomposition_R(Real *R, const Real *F)
    {
        Real C[9];
        matTmatMul3(C, F, F); // C = F^T * F

        Real C2[9];
        matmatTMul3(C2, C, C); // C2 = C * C^T

        Real det = det3(F);

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

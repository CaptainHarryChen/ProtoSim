#pragma once
#include <cuda_utils/block_size.cuh>

namespace cudaPhysics
{
    template <typename Real>
    __global__ void fill_identity_matrix_kernel(Real *dst, unsigned int repeat_time, unsigned int matrix_size)
    {
        unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < repeat_time * matrix_size * matrix_size)
        {
            unsigned int id_in_mat = index % (matrix_size * matrix_size);
            unsigned int row = id_in_mat / matrix_size;
            unsigned int col = id_in_mat % matrix_size;
            dst[index] = (row == col) ? 1 : 0;
        }
    }
    template <typename Real>
    void fill_identity_matrix(Real *dst, unsigned int repeat_time, unsigned int matrix_size)
    {
        fill_identity_matrix_kernel<Real><<<CUDA_GRID_SIZE(repeat_time * matrix_size * matrix_size), CUDA_BLOCK_SIZE>>>(dst, repeat_time, matrix_size);
    }

    template <typename Real>
    __global__ void array_real_inv_kernel(Real *dst, Real *src, unsigned int size)
    {
        unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size)
        {
            dst[index] = 1.0f / src[index];
        }
    }
    template <typename Real>
    void array_real_inv(Real *dst, Real *src, unsigned int size)
    {
        array_real_inv_kernel<Real><<<CUDA_GRID_SIZE(size), CUDA_BLOCK_SIZE>>>(dst, src, size);
    }
}

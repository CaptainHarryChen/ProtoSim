#pragma once
#include <cuda_utils/block_size.cuh>

namespace cuda_utils
{
    template <typename Real>
    __global__ void fill_array_kernel(Real *dst, size_t dst_size, const Real value[], size_t value_size)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < dst_size)
            dst[index] = value[index % value_size];
    }

    template <typename Real>
    void fill_array(Real *dst, size_t dst_size, const Real value[], size_t value_size)
    {
        fill_array_kernel<Real><<<CUDA_GRID_SIZE(dst_size), CUDA_BLOCK_SIZE>>>(dst, dst_size, value, value_size);
    }
}

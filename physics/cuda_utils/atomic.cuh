#pragma once

namespace cudaPhysics
{
    template <typename Real>
    __device__ inline void AtomicMin(Real *addr, Real value)
    {
        if constexpr (std::is_same_v<Real, float>)
        {
            (value >= 0) ? atomicMin((int *)addr, __float_as_int(value)) : atomicMax((unsigned int *)addr, __float_as_uint(value));
        }
        else if constexpr (std::is_same_v<Real, double>)
        {
            if(value >= 0)
            {
                atomicMin((long long int *)addr, __double_as_longlong(value));
            }
            else
            {
                unsigned long long int uval = (*(unsigned long long int*)(&value));
                atomicMax((unsigned long long int *)addr, uval);
            }
        }
        else
        {
            static_assert(false, "Unsupported type for atomicMin");
        }
    }

    template <typename Real>
    __device__ inline void AtomicMax(Real *addr, Real value)
    {
        if constexpr (std::is_same_v<Real, float>)
        {
            (value >= 0) ? atomicMax((int *)addr, __float_as_int(value)) : atomicMin((unsigned int *)addr, __float_as_uint(value));
        }
        else if constexpr (std::is_same_v<Real, double>)
        {
            if(value >= 0)
            {
                atomicMax((long long int *)addr, __double_as_longlong(value));
            }
            else
            {
                unsigned long long int uval = (*(unsigned long long int*)(&value));
                atomicMin((unsigned long long int *)addr, uval);
            }
        }
        else
        {
            static_assert(false, "Unsupported type for atomicMax");
        }
    }
}

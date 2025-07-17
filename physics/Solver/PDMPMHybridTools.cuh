#pragma once
#include "PDMPMHybridSolver.cuh"

namespace PDMPMHybridTools
{
    __host__ __device__ __forceinline__ uint64_t pack_tri_info(float closest_distance, bool inside, unsigned int tri_idx)
    {
        uint64_t result = 0;
        uint32_t *result_ptr = (uint32_t *)&result;
        result_ptr[0] = *((uint32_t *)&closest_distance);
        result_ptr[1] = tri_idx;
        result_ptr[1] |= (inside ? 1u << 31 : 0u);
        return result;
    }

    __host__ __device__ __forceinline__ void unpack_tri_info(uint64_t packed_info, float &closest_distance, bool &inside, unsigned int &tri_idx)
    {
        uint32_t *info_ptr = (uint32_t *)&packed_info;
        closest_distance = *((float *)&info_ptr[0]);
        tri_idx = info_ptr[1] & 0x7FFFFFFFu; // Clear the inside bit
        inside = (info_ptr[1] & (1u << 31)) != 0;
        if (packed_info == 0xFFFFFFFFFFFFFFFFllu)
            inside = false;
    }

    template <typename Real>
    __device__ void get_grid_position(Real *grid_position, unsigned int id, const PDMPMHybridSolverData<Real> *data)
    {
        unsigned int z = id % data->dev_grid_size[2];
        unsigned int y = (id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int x = id / data->dev_grid_size[2] / data->dev_grid_size[1];
        grid_position[0] = data->dev_outer_bbox[0] + x * data->m_grid_spacing;
        grid_position[1] = data->dev_outer_bbox[1] + y * data->m_grid_spacing;
        grid_position[2] = data->dev_outer_bbox[2] + z * data->m_grid_spacing;
    }

    template <typename Real>
    __device__ Real grid_particle_quadratic_weight(const Real *grid_position, const Real *particle_position, Real grid_spacing)
    {
        Real result = 1.;
        for (unsigned int i = 0; i < 3; ++i)
        {
            Real d = (particle_position[i] - grid_position[i]) / grid_spacing;
            Real w = 0;
            if (-0.5 < d && d < 0.5)
                w = 0.75 - d * d;
            else if (0.5 <= d && d < 1.5)
                w = 0.5 * (1.5 - d) * (1.5 - d);
            else if (-1.5 < d && d <= -0.5)
                w = 0.5 * (1.5 + d) * (1.5 + d);
            result *= w;
        }
        return result;
    }
}

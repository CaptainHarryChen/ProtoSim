#include "SeparatingCorrector.cuh"
#include <cuda_utils/cuda_utils.cuh>
#include <Math/algebra.cuh>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

namespace SeparatingCorrectorKernel
{
    template <typename Real>
    __global__ void calc_radius(Real *dev_particle_volume, Real *dev_particle_radius, unsigned int num_particle)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= num_particle)
            return;
        if constexpr (std::is_same_v<Real, float>)
            dev_particle_radius[i] = cbrtf((3.0f / 4.0f) * dev_particle_volume[i] / 3.14159265f);
        else if constexpr (std::is_same_v<Real, double>)
            dev_particle_radius[i] = cbrt((3.0f / 4.0f) * dev_particle_volume[i] / 3.14159265);
    }

    template <typename Real>
    __global__ void calc_particle_to_grid_id(SeparatingCorrectorData<Real> data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data.m_num_particle)
            return;
        data.dev_particle_id[i] = i;
        // find the nearest grid, so plus 0.5 and floor
        int x = floor((data.dev_particle_position[i * 3 + 0] - data.m_outer_bbox[0]) / data.m_grid_spacing + 0.5);
        int y = floor((data.dev_particle_position[i * 3 + 1] - data.m_outer_bbox[1]) / data.m_grid_spacing + 0.5);
        int z = floor((data.dev_particle_position[i * 3 + 2] - data.m_outer_bbox[2]) / data.m_grid_spacing + 0.5);
        if (x < 0 || y < 0 || z < 0 || x >= data.m_grid_size[0] || y >= data.m_grid_size[1] || z >= data.m_grid_size[2])
        {
            x = max(0, min(x, (int)data.m_grid_size[0] - 3));
            y = max(0, min(y, (int)data.m_grid_size[1] - 3));
            z = max(0, min(z, (int)data.m_grid_size[2] - 3));
            printf("Corrector Warning: particle %d is out of grid, set to %d %d %d [Particle position] %.10f %.10f %.10f\n", i, x, y, z, data.dev_particle_position[i * 3 + 0], data.dev_particle_position[i * 3 + 1], data.dev_particle_position[i * 3 + 2]);
        }
        int idx = x * data.m_grid_size[1] * data.m_grid_size[2] + y * data.m_grid_size[2] + z;
        data.dev_particle_to_grid_id[i] = idx;
        atomicAdd(&data.dev_grid_bin_size[idx], 1u);
    }

    template <typename Real>
    __global__ void calc_particle_delta_position(SeparatingCorrectorData<Real> data)
    {
        unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= data.m_num_particle)
            return;
        const Real *particle_position = &data.dev_particle_position[p * 3];
        Real res[3] = {0};
        unsigned int denorm = 0;
        unsigned int center_grid_id = data.dev_particle_to_grid_id[p];
        int x = center_grid_id / data.m_grid_size[1] / data.m_grid_size[2];
        int y = (center_grid_id / data.m_grid_size[2]) % data.m_grid_size[1];
        int z = center_grid_id % data.m_grid_size[2];
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dz = -1; dz <= 1; ++dz)
                {
                    if (x + dx < 0 || y + dy < 0 || z + dz < 0 || x + dx >= data.m_grid_size[0] || y + dy >= data.m_grid_size[1] || z + dz >= data.m_grid_size[2])
                        continue;
                    unsigned int grid_id = (x + dx) * data.m_grid_size[1] * data.m_grid_size[2] + (y + dy) * data.m_grid_size[2] + (z + dz);
                    for (int i = data.dev_grid_bin_start_idx[grid_id]; i < data.dev_grid_bin_start_idx[grid_id + 1]; ++i)
                    {
                        unsigned int np = data.dev_particle_id[i];
                        if (np == p)
                            continue;
                        const Real *np_position = &data.dev_particle_position[np * 3];
                        Real dir[3];
                        cudaPhysics::vecSubs3(dir, particle_position, np_position);
                        Real dist = cudaPhysics::len3(dir);
                        Real min_dist = data.dev_particle_radius[p] + data.dev_particle_radius[np];
                        if (dist < min_dist && dist > 1e-6)
                        {
                            cudaPhysics::axpby(res, (Real)1, res, (min_dist - dist) / dist * 0.5f, dir, 3);
                            denorm += 1;
                        }
                    }
                }
        cudaPhysics::vecMul3(&data.dev_particle_delta_position[p * 3], denorm > 0 ? (Real)1.0f / denorm : (Real)0, res);
    }

    template <typename Real>
    __global__ void particle_box_constraint(SeparatingCorrectorData<Real> data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data.m_num_particle)
            return;
        for (unsigned int d = 0; d < 3; ++d)
        {
            if (data.dev_particle_position[i * 3 + d] < data.m_inner_bbox[d])
                data.dev_particle_position[i * 3 + d] = data.m_inner_bbox[d];
            if (data.dev_particle_position[i * 3 + d] > data.m_inner_bbox[d + 3])
                data.dev_particle_position[i * 3 + d] = data.m_inner_bbox[d + 3];
        }
    }
}

template <typename Real>
SeparatingCorrector<Real>::SeparatingCorrector(
    unsigned int num_particle,
    Real *dev_particle_position,
    Real *dev_particle_volume,
    std::vector<Real> bbox,
    Real grid_spacing,
    unsigned int boundary_thickness,
    unsigned int max_iteration
)
{
    m_data.m_num_particle = num_particle;
    m_data.dev_particle_position = dev_particle_position;
    m_data.dev_particle_volume = dev_particle_volume;
    m_max_iteration = max_iteration;

    // need expand the bbox to ensure the particles can get 3x3x3 grids
    assert(bbox.size() == 6);
    m_data.m_grid_spacing = grid_spacing;
    m_data.m_boundary_thickness = boundary_thickness;
    for (unsigned int i = 0; i < 6; ++i)
    {
        m_data.m_inner_bbox[i] = bbox[i];
        m_data.m_outer_bbox[i] = bbox[i];
    }
    for (unsigned int i = 0; i < 3; ++i)
    {
        m_data.m_outer_bbox[i] -= m_data.m_boundary_thickness * m_data.m_grid_spacing;
        m_data.m_outer_bbox[i + 3] += m_data.m_boundary_thickness * m_data.m_grid_spacing;
    }
    for (unsigned int i = 0; i < 3; ++i)
        m_data.m_grid_size[i] = (unsigned int)(floor((m_data.m_outer_bbox[i + 3] - m_data.m_outer_bbox[i]) / m_data.m_grid_spacing + 0.5)) + 1;
    printf("SeparatingCorrector: grid_size = [%u %u %u]\n", m_data.m_grid_size[0], m_data.m_grid_size[1], m_data.m_grid_size[2]);
    for (unsigned int i = 0; i < 3; ++i)
        m_data.m_outer_bbox[i + 3] = m_data.m_outer_bbox[i] + m_data.m_grid_spacing * m_data.m_grid_size[i];
    m_data.m_num_grid = m_data.m_grid_size[0] * m_data.m_grid_size[1] * m_data.m_grid_size[2];
    printf("SeparatingCorrector: num_grid = %u\n", m_data.m_num_grid);

    cudaMalloc((void **)&m_data.dev_particle_to_grid_id, sizeof(unsigned int) * m_data.m_num_particle);
    cudaMalloc((void **)&m_data.dev_particle_radius, sizeof(Real) * m_data.m_num_particle);
    cudaMalloc((void **)&m_data.dev_particle_delta_position, sizeof(Real) * m_data.m_num_particle * 3);
    cudaMalloc((void **)&m_data.dev_particle_id, sizeof(unsigned int) * m_data.m_num_particle);
    cudaMalloc((void **)&m_data.dev_grid_bin_start_idx, sizeof(unsigned int) * (m_data.m_num_grid + 1));
    cudaMalloc((void **)&m_data.dev_grid_bin_size, sizeof(unsigned int) * m_data.m_num_grid);

    SeparatingCorrectorKernel::calc_radius<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_data.dev_particle_volume, m_data.dev_particle_radius, m_data.m_num_particle);

    cudaCheck(cudaDeviceSynchronize());
    printf("SeparatingCorrector initialized.\n");
}

template <typename Real>
SeparatingCorrector<Real>::~SeparatingCorrector()
{
    cudaFree(m_data.dev_particle_to_grid_id);
    cudaFree(m_data.dev_particle_radius);
    cudaFree(m_data.dev_particle_delta_position);
    cudaFree(m_data.dev_particle_id);
    cudaFree(m_data.dev_grid_bin_start_idx);
    cudaFree(m_data.dev_grid_bin_size);
}

template <typename Real>
void SeparatingCorrector<Real>::Run()
{
    for (unsigned int iter = 0; iter < m_max_iteration; ++iter)
    {
        CalculateParticleDeltaPosition();
        thrust::transform(
            thrust::device,
            m_data.dev_particle_position,
            m_data.dev_particle_position + m_data.m_num_particle * 3,
            m_data.dev_particle_delta_position,
            m_data.dev_particle_position,
            thrust::placeholders::_1 + thrust::placeholders::_2
        );
        ApplyBoxConstraint();
    }
}

template <typename Real>
void SeparatingCorrector<Real>::CalculateParticleDeltaPosition()
{
    cudaMemset(m_data.dev_grid_bin_size, 0, sizeof(unsigned int) * m_data.m_num_grid);
    SeparatingCorrectorKernel::calc_particle_to_grid_id<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_data);
    thrust::sort_by_key(
        thrust::device,
        m_data.dev_particle_to_grid_id,
        m_data.dev_particle_to_grid_id + m_data.m_num_particle,
        m_data.dev_particle_id
    );
    thrust::exclusive_scan(
        thrust::device,
        m_data.dev_grid_bin_size,
        m_data.dev_grid_bin_size + m_data.m_num_grid,
        m_data.dev_grid_bin_start_idx + 1
    );
    SeparatingCorrectorKernel::calc_particle_delta_position<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_data);
}

template <typename Real>
void SeparatingCorrector<Real>::ApplyBoxConstraint()
{
    SeparatingCorrectorKernel::particle_box_constraint<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_data);
}

template class SeparatingCorrector<float>;
template class SeparatingCorrector<double>;
template struct SeparatingCorrectorData<float>;
template struct SeparatingCorrectorData<double>;
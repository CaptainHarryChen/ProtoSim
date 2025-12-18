#include "PICVolumeCorrector.cuh"
#include <cuda_utils/cuda_utils.cuh>
#include <Math/algebra.cuh>
#include "CPICTools.cuh"
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

namespace PICVolumeCorrectorKernel
{
    template <typename Real>
    __global__ void calc_particle_to_leftbottom_grid_id(PICVolumeCorrectorData<Real> data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data.m_num_particle)
            return;
        // find the nearest grid, so plus 0.5 and floor
        int x = floor((data.dev_particle_position[i * 3 + 0] - data.m_outer_bbox[0]) / data.m_grid_spacing + 0.5);
        int y = floor((data.dev_particle_position[i * 3 + 1] - data.m_outer_bbox[1]) / data.m_grid_spacing + 0.5);
        int z = floor((data.dev_particle_position[i * 3 + 2] - data.m_outer_bbox[2]) / data.m_grid_spacing + 0.5);
        // get the left bottom grid id
        x--;
        y--;
        z--;
        if (x < 0 || y < 0 || z < 0 || x >= data.m_grid_size[0] - 2 || y >= data.m_grid_size[1] - 2 || z >= data.m_grid_size[2] - 2)
        {
            x = max(0, min(x, (int)data.m_grid_size[0] - 3));
            y = max(0, min(y, (int)data.m_grid_size[1] - 3));
            z = max(0, min(z, (int)data.m_grid_size[2] - 3));
            printf("Warning: particle %d is out of grid, set to %d %d %d [Particle position] %.10f %.10f %.10f\n", i, x, y, z, data.dev_particle_position[i * 3 + 0], data.dev_particle_position[i * 3 + 1], data.dev_particle_position[i * 3 + 2]);
        }

        data.dev_particle_to_grid_id[i] = x * data.m_grid_size[1] * data.m_grid_size[2] + y * data.m_grid_size[2] + z;
    }

    template <typename Real>
    __global__ void P2G_grid_volume(PICVolumeCorrectorData<Real> data)
    {
        unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= data.m_num_particle)
            return;
        const Real *particle_position = &data.dev_particle_position[p * 3];
        unsigned int leftbottom_grid_id = data.dev_particle_to_grid_id[p];
        unsigned int x = leftbottom_grid_id / data.m_grid_size[1] / data.m_grid_size[2];
        unsigned int y = (leftbottom_grid_id / data.m_grid_size[2]) % data.m_grid_size[1];
        unsigned int z = leftbottom_grid_id % data.m_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data.m_grid_size[1] * data.m_grid_size[2] + (y + dy) * data.m_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    CPICTools::get_grid_position(grid_position, grid_id, data.m_grid_size, data.m_outer_bbox, data.m_grid_spacing);
                    Real weight = CPICTools::grid_particle_quadratic_weight(grid_position, particle_position, data.m_grid_spacing);
                    atomicAdd(&data.dev_grid_volume[grid_id], data.dev_particle_volume[p] * weight);
                }
    }

    template <typename Real>
    __global__ void G2P_particle_delta_position(PICVolumeCorrectorData<Real> data)
    {
        unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= data.m_num_particle)
            return;
        const Real *particle_position = &data.dev_particle_position[p * 3];
        Real res[3] = {0};
        unsigned int leftbottom_grid_id = data.dev_particle_to_grid_id[p];
        unsigned int x = leftbottom_grid_id / data.m_grid_size[1] / data.m_grid_size[2];
        unsigned int y = (leftbottom_grid_id / data.m_grid_size[2]) % data.m_grid_size[1];
        unsigned int z = leftbottom_grid_id % data.m_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data.m_grid_size[1] * data.m_grid_size[2] + (y + dy) * data.m_grid_size[2] + (z + dz);
                    Real grid_position[3], delta_position[3];
                    CPICTools::get_grid_position(grid_position, grid_id, data.m_grid_size, data.m_outer_bbox, data.m_grid_spacing);
                    cudaPhysics::vecSubs3(delta_position, grid_position, particle_position);
                    Real coef = 1.0f - data.dev_grid_volume[grid_id] / data.m_grid_spacing / data.m_grid_spacing / data.m_grid_spacing;
                    coef = min(max(coef, -0.5f), 0.5f);
                    coef *= CPICTools::grid_particle_quadratic_weight(grid_position, particle_position, data.m_grid_spacing);
                    cudaPhysics::axpby(res, (Real)1, res, coef, delta_position, 3);
                }
        cudaPhysics::vecCopy3(&data.dev_particle_delta_position[p * 3], res);
    }

    template <typename Real>
    __global__ void particle_box_constraint(PICVolumeCorrectorData<Real> data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data.m_num_particle)
            return;
        for (unsigned int d = 0; d < 3; ++d)
        {
            Real new_pos = data.dev_particle_position[i * 3 + d] + data.dev_particle_delta_position[i * 3 + d];
            if (new_pos < data.m_inner_bbox[d])
                new_pos = data.m_inner_bbox[d];
            if (new_pos > data.m_inner_bbox[d + 3])
                new_pos = data.m_inner_bbox[d + 3];
            data.dev_particle_delta_position[i * 3 + d] = new_pos - data.dev_particle_position[i * 3 + d];
        }
    }
}

template <typename Real>
PICVolumeCorrector<Real>::PICVolumeCorrector(
    unsigned int num_particle,
    Real *dev_particle_position,
    Real *dev_particle_volume,
    unsigned int *dev_particle_to_grid_id,
    std::vector<Real> bbox,
    Real grid_spacing,
    unsigned int boundary_thickness,
    unsigned int max_iteration
)
{
    m_data.m_num_particle = num_particle;
    m_data.dev_particle_position = dev_particle_position;
    m_data.dev_particle_volume = dev_particle_volume;
    m_data.dev_particle_to_grid_id = dev_particle_to_grid_id;   
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
    printf("PICVolumeCorrector: grid_size = [%u %u %u]\n", m_data.m_grid_size[0], m_data.m_grid_size[1], m_data.m_grid_size[2]);
    for (unsigned int i = 0; i < 3; ++i)
        m_data.m_outer_bbox[i + 3] = m_data.m_outer_bbox[i] + m_data.m_grid_spacing * m_data.m_grid_size[i];
    m_data.m_num_grid = m_data.m_grid_size[0] * m_data.m_grid_size[1] * m_data.m_grid_size[2];
    printf("PICVolumeCorrector: num_grid = %u\n", m_data.m_num_grid);

    cudaMalloc((void **)&m_data.dev_particle_delta_position, sizeof(Real) * m_data.m_num_particle * 3);
    cudaMalloc((void **)&m_data.dev_grid_volume, sizeof(Real) * m_data.m_num_grid);

    cudaCheck(cudaDeviceSynchronize());
    printf("PICVolumeCorrector initialized.\n");
}

template <typename Real>
PICVolumeCorrector<Real>::~PICVolumeCorrector()
{
    cudaFree(m_data.dev_particle_delta_position);
    cudaFree(m_data.dev_grid_volume);
}

template <typename Real>
void PICVolumeCorrector<Real>::Run()
{
    for (unsigned int iter = 0; iter < m_max_iteration; ++iter)
    {
        CalculateParticleDeltaPosition();
        ApplyBoxConstraint();
        thrust::transform(
            thrust::device_pointer_cast(m_data.dev_particle_position),
            thrust::device_pointer_cast(m_data.dev_particle_position) + m_data.m_num_particle * 3,
            thrust::device_pointer_cast(m_data.dev_particle_delta_position),
            thrust::device_pointer_cast(m_data.dev_particle_position),
            thrust::placeholders::_1 + thrust::placeholders::_2
        );
    }
}

template <typename Real>
void PICVolumeCorrector<Real>::CalculateParticleDeltaPosition()
{
    PICVolumeCorrectorKernel::calc_particle_to_leftbottom_grid_id<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_data);
    cudaMemset(m_data.dev_grid_volume, 0, sizeof(Real) * m_data.m_num_grid);
    PICVolumeCorrectorKernel::P2G_grid_volume<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_data);
    PICVolumeCorrectorKernel::G2P_particle_delta_position<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_data);
}

template <typename Real>
void PICVolumeCorrector<Real>::ApplyBoxConstraint()
{
    PICVolumeCorrectorKernel::particle_box_constraint<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_data);
}

template class PICVolumeCorrector<float>;
template class PICVolumeCorrector<double>;
template struct PICVolumeCorrectorData<float>;
template struct PICVolumeCorrectorData<double>;
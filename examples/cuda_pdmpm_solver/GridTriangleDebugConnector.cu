#include "GridTriangleDebugConnector.cuh"
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <cuda_utils/error.cuh>
#include <cuda_utils/block_size.cuh>
#include <Geometry/ParticleBatch.h>
#include <Solver/PDMPMHybridSolver.cuh>
#include <Solver/PDMPMHybridTools.cuh>

namespace GridTriangleDebugConnectorKernel
{
    template <typename Real>
    __global__ void transfer_data(Particle *dev_particles, PDMPMHybridSolverData<Real> *data, unsigned int num_particle)
    {
        unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= num_particle)
            return;
        if (data->dev_grid_tri_info[id] == 0xFFFFFFFFFFFFFFFFllu)
        {
            dev_particles[id].Position = glm::vec3(10000.0f, 10000.0f, 10000.0f);
            dev_particles[id].Color = glm::vec3(1.0f, 1.0f, 1.0f);
            return;
        }
        unsigned int z = id % data->dev_grid_size[2];
        unsigned int y = (id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int x = id / data->dev_grid_size[2] / data->dev_grid_size[1];
        dev_particles[id].Position.x = data->dev_outer_bbox[0] + x * data->m_grid_spacing;
        dev_particles[id].Position.y = data->dev_outer_bbox[1] + y * data->m_grid_spacing;
        dev_particles[id].Position.z = data->dev_outer_bbox[2] + z * data->m_grid_spacing;
        float dis;
        bool inside;
        unsigned int idx;
        PDMPMHybridTools::unpack_tri_info(data->dev_grid_tri_info[id], dis, inside, idx);
        if (inside)
        {
            dev_particles[id].Color = glm::vec3(1.0f, 0.5f, 0.0f);
        }
        else
        {
            dev_particles[id].Color = glm::vec3(0.0f, 1.0f, 0.0f);
        }
    }
}

template <typename Real>
GridTriangleDebugConnector<Real>::GridTriangleDebugConnector(std::shared_ptr<ParticleBatch> particles, PDMPMHybridSolverData<Real> *data, PDMPMHybridSolverData<Real> *dev_data)
    : m_particles(particles), m_data(data), m_dev_data(dev_data)
{
    cudaCheck(cudaGraphicsGLRegisterBuffer(&m_cuda_resource_buf, particles->GetVBO(), cudaGraphicsRegisterFlagsNone));
}

template <typename Real>
GridTriangleDebugConnector<Real>::~GridTriangleDebugConnector()
{
    // cudaCheck(cudaGraphicsUnregisterResource(m_cuda_resource_buf));
    cudaFree(m_dev_data);
}

template <typename Real>
void GridTriangleDebugConnector<Real>::TransferData()
{
    cudaCheck(cudaGraphicsMapResources(1, &m_cuda_resource_buf));
    Particle *dev_particles;
    size_t buffer_size;
    cudaCheck(cudaGraphicsResourceGetMappedPointer((void **)&dev_particles, &buffer_size, m_cuda_resource_buf));
    unsigned int num_particle = (unsigned int)(buffer_size / sizeof(Particle));
    assert(num_particle == m_data->m_num_grid);
    GridTriangleDebugConnectorKernel::transfer_data<Real><<<CUDA_GRID_SIZE(num_particle), CUDA_BLOCK_SIZE>>>(dev_particles, m_dev_data, num_particle);
    cudaCheck(cudaGraphicsUnmapResources(1, &m_cuda_resource_buf));
}

template class GridTriangleDebugConnector<float>;
template class GridTriangleDebugConnector<double>;

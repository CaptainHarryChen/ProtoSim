#include "ParticleConnector.cuh"
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <cuda_utils/error.cuh>
#include <cuda_utils/block_size.cuh>
#include <Geometry/ParticleBatch.h>

namespace ParticleConnectorKernel
{
    template <typename Real>
    __global__ void transfer_data(Particle *dev_particles, Real *dev_position, Real *dev_color, unsigned int num_particles)
    {
        const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= num_particles)
            return;
        if (dev_position)
        {
            dev_particles[index].Position.x = dev_position[index * 3 + 0];
            dev_particles[index].Position.y = dev_position[index * 3 + 1];
            dev_particles[index].Position.z = dev_position[index * 3 + 2];
        }
        if (dev_color)
        {
            dev_particles[index].Color.x = dev_color[index * 3 + 0];
            dev_particles[index].Color.y = dev_color[index * 3 + 1];
            dev_particles[index].Color.z = dev_color[index * 3 + 2];
        }
    }
}

template <typename Real>
ParticleConnector<Real>::ParticleConnector(std::shared_ptr<ParticleBatch> particle_batch, Real *dev_position, Real *dev_color)
    : m_particle_batch(particle_batch), m_dev_position(dev_position), m_dev_color(dev_color)
{
    cudaCheck(cudaGraphicsGLRegisterBuffer(&m_cuda_resource_buf, particle_batch->m_VBO, cudaGraphicsRegisterFlagsNone));
}

template <typename Real>
ParticleConnector<Real>::~ParticleConnector()
{
    // BUG: cudaErrorInvalidGraphicsContext
    // cudaCheck(cudaGraphicsUnregisterResource(m_cuda_resource_buf));
}

template <typename Real>
void ParticleConnector<Real>::TransferData()
{
    cudaCheck(cudaGraphicsMapResources(1, &m_cuda_resource_buf));
    Particle *dev_particles;
    size_t buffer_size;
    cudaCheck(cudaGraphicsResourceGetMappedPointer((void **)&dev_particles, &buffer_size, m_cuda_resource_buf));
    unsigned int num_particles = (unsigned int)(buffer_size / sizeof(Particle));
    ParticleConnectorKernel::transfer_data<Real><<<CUDA_GRID_SIZE(num_particles), CUDA_BLOCK_SIZE>>>(dev_particles, m_dev_position, m_dev_color, num_particles);
    cudaCheck(cudaGraphicsUnmapResources(1, &m_cuda_resource_buf));
}

template class ParticleConnector<float>;
template class ParticleConnector<double>;

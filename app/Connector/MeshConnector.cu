#include "MeshConnector.cuh"
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <cuda_utils/error.cuh>
#include <cuda_utils/block_size.cuh>
#include <Mesh/Mesh.h>

namespace MeshConnectorKernel
{
    template <typename Real>
    __global__ void transfer_data(Vertex *dev_vertices, Real *dev_position, unsigned int num_vertices)
    {
        const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= num_vertices)
            return;
        dev_vertices[index].position.x = dev_position[index * 3 + 0];
        dev_vertices[index].position.y = dev_position[index * 3 + 1];
        dev_vertices[index].position.z = dev_position[index * 3 + 2];
    }
}

template <typename Real>
MeshConnector<Real>::MeshConnector(std::shared_ptr<Mesh> mesh, Real *dev_position)
    : m_mesh(mesh), m_dev_position(dev_position)
{
    cudaCheck(cudaGraphicsGLRegisterBuffer(&m_cuda_resource_buf, mesh->m_VBO, cudaGraphicsRegisterFlagsNone));
}

template <typename Real>
MeshConnector<Real>::~MeshConnector()
{
    // BUG: cudaErrorInvalidGraphicsContext
    // cudaCheck(cudaGraphicsUnregisterResource(m_cuda_resource_buf));
}

template <typename Real>
void MeshConnector<Real>::TransferData()
{
    cudaCheck(cudaGraphicsMapResources(1, &m_cuda_resource_buf));
    Vertex *dev_vertices;
    size_t buffer_size;
    cudaCheck(cudaGraphicsResourceGetMappedPointer((void **)&dev_vertices, &buffer_size, m_cuda_resource_buf));
    unsigned int num_vertices = (unsigned int)(buffer_size / sizeof(Vertex));
    MeshConnectorKernel::transfer_data<Real><<<CUDA_GRID_SIZE(num_vertices), CUDA_BLOCK_SIZE>>>(dev_vertices, m_dev_position, num_vertices);
    cudaCheck(cudaGraphicsUnmapResources(1, &m_cuda_resource_buf));
}

template class MeshConnector<float>;
template class MeshConnector<double>;

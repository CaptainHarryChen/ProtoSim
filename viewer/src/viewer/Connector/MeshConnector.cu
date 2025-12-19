#include "MeshConnector.cuh"
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <viewer/RenderObject/Mesh.h>

namespace MeshConnectorKernel {
template <typename Real>
__global__ void transfer_data(viewer::Vertex* dev_vertices, Real* dev_position, unsigned int num_vertices)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_vertices)
        return;
    dev_vertices[index].position.x = dev_position[index * 3 + 0];
    dev_vertices[index].position.y = dev_position[index * 3 + 1];
    dev_vertices[index].position.z = dev_position[index * 3 + 2];
}
}  // namespace MeshConnectorKernel

namespace viewer {

template <typename Real>
MeshConnector<Real>::MeshConnector(std::shared_ptr<Mesh> mesh, Real* dev_position)
    : m_mesh(mesh), m_dev_position(dev_position)
{
    cudaGraphicsGLRegisterBuffer(&m_cuda_resource_buf, mesh->m_VBO, cudaGraphicsRegisterFlagsNone);
}

template <typename Real>
MeshConnector<Real>::~MeshConnector()
{
    // BUG: cudaErrorInvalidGraphicsContext
    // cudaGraphicsUnregisterResource(m_cuda_resource_buf);
}

template <typename Real>
void MeshConnector<Real>::TransferData()
{
    cudaGraphicsMapResources(1, &m_cuda_resource_buf);
    Vertex* dev_vertices;
    size_t  buffer_size;
    cudaGraphicsResourceGetMappedPointer(( void** )&dev_vertices, &buffer_size, m_cuda_resource_buf);
    unsigned int num_vertices = ( unsigned int )(buffer_size / sizeof(Vertex));
    MeshConnectorKernel::transfer_data<Real>
        <<<(num_vertices + VIEWER_CONNECTOR_CUDA_BLOCK_SIZE - 1) / VIEWER_CONNECTOR_CUDA_BLOCK_SIZE, VIEWER_CONNECTOR_CUDA_BLOCK_SIZE>>>(dev_vertices, m_dev_position, num_vertices);
    cudaGraphicsUnmapResources(1, &m_cuda_resource_buf);
}

template class MeshConnector<float>;
template class MeshConnector<double>;

}  // namespace viewer

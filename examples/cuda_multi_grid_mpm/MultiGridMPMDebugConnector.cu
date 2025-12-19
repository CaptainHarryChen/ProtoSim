#include "MultiGridMPMDebugConnector.cuh"
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <cuda_utils/error.cuh>
#include <cuda_utils/block_size.cuh>
#include <viewer/RenderObject/LineSegment.h>
#include <Solver/MultiGridMPMSolver.cuh>

namespace MultiGridMPMDebugConnectorKernel
{
    template <typename Real>
    __global__ void transfer_data(viewer::LineSeg *dev_lines, MultiGridMPMSolverData<Real> data, unsigned int grid_id, Real scale, unsigned int num_line)
    {
        unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= num_line)
            return;
        unsigned int z = id % data.m_grid_size[2];
        unsigned int y = (id / data.m_grid_size[2]) % data.m_grid_size[1];
        unsigned int x = id / data.m_grid_size[2] / data.m_grid_size[1];
        dev_lines[id].a.x = data.m_outer_bbox[0] + x * data.m_grid_spacing;
        dev_lines[id].a.y = data.m_outer_bbox[1] + y * data.m_grid_spacing;
        dev_lines[id].a.z = data.m_outer_bbox[2] + z * data.m_grid_spacing;
        Real *dev_grid_vector = &data.dev_grid_normal[grid_id * data.m_num_grid / data.m_num_object * 3];
        dev_lines[id].b = dev_lines[id].a + glm::vec3(dev_grid_vector[id * 3 + 0] * scale, dev_grid_vector[id * 3 + 1] * scale, dev_grid_vector[id * 3 + 2] * scale);
    }
}

template <typename Real>
MultiGridMPMDebugConnector<Real>::MultiGridMPMDebugConnector(std::shared_ptr<viewer::LineSegment> lines, MultiGridMPMSolverData<Real> data, unsigned int grid_id, Real scale)
    : m_lines(lines), m_data(data), m_grid_id(grid_id), m_scale(scale)
{
    cudaCheck(cudaGraphicsGLRegisterBuffer(&m_cuda_resource_buf, lines->GetVBO(), cudaGraphicsRegisterFlagsNone));
}

template <typename Real>
MultiGridMPMDebugConnector<Real>::~MultiGridMPMDebugConnector()
{
    // cudaCheck(cudaGraphicsUnregisterResource(m_cuda_resource_buf));
}

template <typename Real>
void MultiGridMPMDebugConnector<Real>::TransferData()
{
    cudaCheck(cudaGraphicsMapResources(1, &m_cuda_resource_buf));
    viewer::LineSeg *dev_lines;
    size_t buffer_size;
    cudaCheck(cudaGraphicsResourceGetMappedPointer((void **)&dev_lines, &buffer_size, m_cuda_resource_buf));
    unsigned int num_lines = (unsigned int)(buffer_size / sizeof(viewer::LineSeg));
    assert(num_lines == m_data.m_num_grid / m_data.m_num_object);
    MultiGridMPMDebugConnectorKernel::transfer_data<Real><<<CUDA_GRID_SIZE(num_lines), CUDA_BLOCK_SIZE>>>(dev_lines, m_data, m_grid_id, m_scale, num_lines);
    cudaCheck(cudaGraphicsUnmapResources(1, &m_cuda_resource_buf));
}

template class MultiGridMPMDebugConnector<float>;
template class MultiGridMPMDebugConnector<double>;

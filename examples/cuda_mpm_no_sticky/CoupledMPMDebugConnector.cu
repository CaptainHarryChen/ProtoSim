#include "CoupledMPMDebugConnector.cuh"
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <cuda_utils/error.cuh>
#include <cuda_utils/block_size.cuh>
#include <Geometry/LineSegment.h>
#include <Solver/CoupledMPMSolver.cuh>

namespace CoupledMPMDebugConnectorKernel
{
    template <typename Real>
    __global__ void transfer_data(LineSeg *dev_lines, CoupledMPMSolverData<Real> *data, unsigned int grid_id, Real scale, unsigned int num_line)
    {
        unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= num_line)
            return;
        unsigned int z = id % data->dev_grid_size[2];
        unsigned int y = (id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int x = id / data->dev_grid_size[2] / data->dev_grid_size[1];
        dev_lines[id].a.x = data->dev_outer_bbox[0] + x * data->m_grid_spacing;
        dev_lines[id].a.y = data->dev_outer_bbox[1] + y * data->m_grid_spacing;
        dev_lines[id].a.z = data->dev_outer_bbox[2] + z * data->m_grid_spacing;
        Real *dev_grid_vector = &data->dev_grid_normal[grid_id * data->m_num_grid / data->m_num_object * 3];
        dev_lines[id].b = dev_lines[id].a + glm::vec3(dev_grid_vector[id * 3 + 0] * scale, dev_grid_vector[id * 3 + 1] * scale, dev_grid_vector[id * 3 + 2] * scale);
    }
}

template <typename Real>
CoupledMPMDebugConnector<Real>::CoupledMPMDebugConnector(std::shared_ptr<LineSegment> lines, CoupledMPMSolverData<Real> *data, unsigned int grid_id, Real scale)
    : m_lines(lines), m_data(data), m_grid_id(grid_id), m_scale(scale)
{
    cudaCheck(cudaGraphicsGLRegisterBuffer(&m_cuda_resource_buf, lines->GetVBO(), cudaGraphicsRegisterFlagsNone));
    cudaMalloc(&m_dev_data, sizeof(CoupledMPMSolverData<Real>));
    cudaMemcpy(m_dev_data, data, sizeof(CoupledMPMSolverData<Real>), cudaMemcpyHostToDevice);
}

template <typename Real>
CoupledMPMDebugConnector<Real>::~CoupledMPMDebugConnector()
{
    // cudaCheck(cudaGraphicsUnregisterResource(m_cuda_resource_buf));
    cudaFree(m_dev_data);
}

template <typename Real>
void CoupledMPMDebugConnector<Real>::TransferData()
{
    cudaCheck(cudaGraphicsMapResources(1, &m_cuda_resource_buf));
    LineSeg *dev_lines;
    size_t buffer_size;
    cudaCheck(cudaGraphicsResourceGetMappedPointer((void **)&dev_lines, &buffer_size, m_cuda_resource_buf));
    unsigned int num_lines = (unsigned int)(buffer_size / sizeof(LineSeg));
    assert(num_lines == m_data->m_num_grid / m_data->m_num_object);
    CoupledMPMDebugConnectorKernel::transfer_data<Real><<<CUDA_GRID_SIZE(num_lines), CUDA_BLOCK_SIZE>>>(dev_lines, m_dev_data, m_grid_id, m_scale, num_lines);
    cudaCheck(cudaGraphicsUnmapResources(1, &m_cuda_resource_buf));
}

template class CoupledMPMDebugConnector<float>;
template class CoupledMPMDebugConnector<double>;

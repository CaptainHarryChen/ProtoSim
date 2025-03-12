#include "EmptySolver.cuh"
#include <cuda_utils/error.cuh>
#include <cuda_utils/block_size.cuh>

namespace EmptySolverKernel
{
    template <typename Real>
    __global__ void step(EmptySolverData<Real> *data)
    {
        const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= data->num_vertices)
            return;
        data->dev_position[index * 3 + 1] -= (Real)0.01;
    }
}

template <typename Real>
EmptySolver<Real>::EmptySolver(const std::vector<Real> &positions)
{
    m_data.num_vertices = (unsigned int)positions.size() / 3;
    cudaCheck(cudaMalloc(&m_data.dev_position, m_data.num_vertices * 3 * sizeof(Real)));
    cudaCheck(cudaMemcpy(m_data.dev_position, positions.data(), m_data.num_vertices * 3 * sizeof(Real), cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&m_dev_data, sizeof(EmptySolverData<Real>)));
    cudaCheck(cudaMemcpy(m_dev_data, &m_data, sizeof(EmptySolverData<Real>), cudaMemcpyHostToDevice));
}

template <typename Real>
EmptySolver<Real>::~EmptySolver()
{
    cudaCheck(cudaMemcpy(&m_data, m_dev_data, sizeof(EmptySolverData<Real>), cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(m_data.dev_position));

    cudaCheck(cudaFree(m_dev_data));
}

template <typename Real>
void EmptySolver<Real>::Step()
{
    EmptySolverKernel::step<Real><<<CUDA_GRID_SIZE(m_data.num_vertices), CUDA_BLOCK_SIZE>>>(m_dev_data);
}

template class EmptySolver<float>;
template class EmptySolver<double>;

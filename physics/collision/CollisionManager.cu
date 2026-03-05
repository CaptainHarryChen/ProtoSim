#include "CollisionManager.cuh"
#include <cuda_utils/cuda_utils.cuh>

template <typename Real>
CollisionManager<Real>::CollisionManager(unsigned int num_bodies, unsigned int max_collisions_per_type)
    : m_num_bodies(num_bodies), m_max_collisions_per_type(max_collisions_per_type)
{
    m_ground_collisions.max_collisions = num_bodies;
    m_body_collisions.max_collisions = max_collisions_per_type;
    m_max_total_collisions = num_bodies + max_collisions_per_type;

    if (m_max_total_collisions == 0)
    {
        m_ground_collisions.dev_collisions = nullptr;
        m_ground_collisions.dev_collision_count = nullptr;
        m_body_collisions.dev_collisions = nullptr;
        m_body_collisions.dev_collision_count = nullptr;
        m_all_collisions = nullptr;
        m_all_collision_count = nullptr;
        return;
    }

    cudaMalloc(&m_ground_collisions.dev_collisions, sizeof(CollisionInfo<Real>) * num_bodies);
    cudaMalloc(&m_ground_collisions.dev_collision_count, sizeof(int));

    cudaMalloc(&m_body_collisions.dev_collisions, sizeof(CollisionInfo<Real>) * max_collisions_per_type);
    cudaMalloc(&m_body_collisions.dev_collision_count, sizeof(int));

    cudaMalloc(&m_all_collisions, sizeof(CollisionInfo<Real>) * m_max_total_collisions);
    cudaMalloc(&m_all_collision_count, sizeof(int));
}

template <typename Real>
CollisionManager<Real>::~CollisionManager()
{
    if (m_ground_collisions.dev_collisions)
        cudaFree(m_ground_collisions.dev_collisions);
    if (m_ground_collisions.dev_collision_count)
        cudaFree(m_ground_collisions.dev_collision_count);
    if (m_body_collisions.dev_collisions)
        cudaFree(m_body_collisions.dev_collisions);
    if (m_body_collisions.dev_collision_count)
        cudaFree(m_body_collisions.dev_collision_count);
    if (m_all_collisions)
        cudaFree(m_all_collisions);
    if (m_all_collision_count)
        cudaFree(m_all_collision_count);
}

template <typename Real>
void CollisionManager<Real>::Clear()
{
    if (m_ground_collisions.dev_collision_count)
    {
        int zero = 0;
        cudaMemcpy(m_ground_collisions.dev_collision_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
    }
    if (m_body_collisions.dev_collision_count)
    {
        int zero = 0;
        cudaMemcpy(m_body_collisions.dev_collision_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
    }
    if (m_all_collision_count)
    {
        int zero = 0;
        cudaMemcpy(m_all_collision_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
    }
}

namespace CollisionManagerKernel
{
    template <typename Real>
    __global__ void merge_collisions_kernel(
        CollisionInfo<Real>* all_collisions,
        int* all_count,
        const CollisionInfo<Real>* ground_collisions,
        const int* ground_count,
        const CollisionInfo<Real>* body_collisions,
        const int* body_count,
        int max_ground,
        int max_body)
    {
        int g_count = *ground_count;
        int b_count = *body_count;

        int i = blockDim.x * blockIdx.x + threadIdx.x;

        if (i < g_count && i < max_ground)
        {
            all_collisions[i] = ground_collisions[i];
        }

        int body_offset = g_count;
        if (i < b_count && i < max_body)
        {
            all_collisions[body_offset + i] = body_collisions[i];
        }

        if (i == 0)
        {
            *all_count = g_count + b_count;
        }
    }
}

template <typename Real>
void CollisionManager<Real>::MergeCollisions()
{
    int ground_count = 0;
    int body_count = 0;
    cudaMemcpy(&ground_count, m_ground_collisions.dev_collision_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&body_count, m_body_collisions.dev_collision_count, sizeof(int), cudaMemcpyDeviceToHost);

    int total_threads = max(ground_count, body_count) + 1;
    int blocks = (total_threads + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    CollisionManagerKernel::merge_collisions_kernel<Real><<<blocks, CUDA_BLOCK_SIZE>>>(
        m_all_collisions,
        m_all_collision_count,
        m_ground_collisions.dev_collisions,
        m_ground_collisions.dev_collision_count,
        m_body_collisions.dev_collisions,
        m_body_collisions.dev_collision_count,
        m_ground_collisions.max_collisions,
        m_body_collisions.max_collisions);
}

template class CollisionManager<float>;
template class CollisionManager<double>;

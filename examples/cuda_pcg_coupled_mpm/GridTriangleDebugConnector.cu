#include "GridTriangleDebugConnector.cuh"
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <cuda_utils/error.cuh>
#include <cuda_utils/block_size.cuh>
#include <viewer/RenderObject/ParticleBatch.h>
#include <viewer/RenderObject/LineSegment.h>
#include <Solver/PCGCoupledMPMSolver.cuh>
#include <Solver/CPICTools.cuh>

namespace GridTriangleDebugConnectorKernel
{
    template <typename Real>
    __global__ void transfer_particle_data(viewer::Particle *dev_particles, PCGCoupledMPMSolverData<Real> *data, unsigned int num_particle)
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
        auto mpm = data->dev_mpm_data;
        unsigned int z = id % mpm->dev_grid_size[2];
        unsigned int y = (id / mpm->dev_grid_size[2]) % mpm->dev_grid_size[1];
        unsigned int x = id / mpm->dev_grid_size[2] / mpm->dev_grid_size[1];
        dev_particles[id].Position.x = mpm->dev_outer_bbox[0] + x * mpm->m_grid_spacing;
        dev_particles[id].Position.y = mpm->dev_outer_bbox[1] + y * mpm->m_grid_spacing;
        dev_particles[id].Position.z = mpm->dev_outer_bbox[2] + z * mpm->m_grid_spacing;
        float dis;
        bool inside;
        unsigned int idx;
        CPICTools::unpack_tri_info(data->dev_grid_tri_info[id], dis, inside, idx);
        if (inside)
        {
            dev_particles[id].Color = glm::vec3(1.0f, 0.5f, 0.0f);
        }
        else
        {
            dev_particles[id].Color = glm::vec3(0.0f, 1.0f, 0.0f);
        }
    }

    template <typename Real>
    __global__ void transfer_line_seg_data(viewer::LineSeg *dev_line_segs, PCGCoupledMPMSolverData<Real> *data, unsigned int num_particle)
    {
        unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= num_particle)
            return;
        if (data->dev_grid_tri_info[id] == 0xFFFFFFFFFFFFFFFFllu)
        {
            dev_line_segs[id].a = glm::vec3(0.0f);
            dev_line_segs[id].b = glm::vec3(0.0f);
            return;
        }
        auto mpm = data->dev_mpm_data;
        unsigned int z = id % mpm->dev_grid_size[2];
        unsigned int y = (id / mpm->dev_grid_size[2]) % mpm->dev_grid_size[1];
        unsigned int x = id / mpm->dev_grid_size[2] / mpm->dev_grid_size[1];
        dev_line_segs[id].a.x = mpm->dev_outer_bbox[0] + x * mpm->m_grid_spacing;
        dev_line_segs[id].a.y = mpm->dev_outer_bbox[1] + y * mpm->m_grid_spacing;
        dev_line_segs[id].a.z = mpm->dev_outer_bbox[2] + z * mpm->m_grid_spacing;
        float dis;
        bool inside;
        unsigned int idx;
        CPICTools::unpack_tri_info(data->dev_grid_tri_info[id], dis, inside, idx);
        auto fem = data->dev_fem_data;
        const unsigned int *tri = &data->dev_triangle[idx * 3];
        const Real *tri_a_pos = &fem->dev_vert_position[tri[0] * 3];
        const Real *tri_b_pos = &fem->dev_vert_position[tri[1] * 3];
        const Real *tri_c_pos = &fem->dev_vert_position[tri[2] * 3];
        dev_line_segs[id].b.x = (tri_a_pos[0] + tri_b_pos[0] + tri_c_pos[0]) / 3.0f;
        dev_line_segs[id].b.y = (tri_a_pos[1] + tri_b_pos[1] + tri_c_pos[1]) / 3.0f;
        dev_line_segs[id].b.z = (tri_a_pos[2] + tri_b_pos[2] + tri_c_pos[2]) / 3.0f;

        // float cur_dis = glm::length(glm::vec3(dev_line_segs[id].b.x - dev_line_segs[id].a.x,
        //                               dev_line_segs[id].b.y - dev_line_segs[id].a.y,
        //                               dev_line_segs[id].b.z - dev_line_segs[id].a.z));
        // if (cur_dis <= 0.4f)
        // {
        //     dev_line_segs[id].a = glm::vec3(0.0f);
        //     dev_line_segs[id].b = glm::vec3(0.0f);
        // }
    }
}

template <typename Real>
GridTriangleDebugConnector<Real>::GridTriangleDebugConnector(std::shared_ptr<viewer::ParticleBatch> particles, std::shared_ptr<viewer::LineSegment> line_segs, PCGCoupledMPMSolverData<Real> *data, PCGCoupledMPMSolverData<Real> *dev_data)
    : m_particles(particles), m_line_segs(line_segs), m_data(data), m_dev_data(dev_data)
{
    cudaCheck(cudaGraphicsGLRegisterBuffer(&m_particle_buf, particles->GetVBO(), cudaGraphicsRegisterFlagsNone));
    cudaCheck(cudaGraphicsGLRegisterBuffer(&m_line_seg_buf, line_segs->GetVBO(), cudaGraphicsRegisterFlagsNone));
}

template <typename Real>
GridTriangleDebugConnector<Real>::~GridTriangleDebugConnector()
{
    // cudaCheck(cudaGraphicsUnregisterResource(m_cuda_resource_buf));
}

template <typename Real>
void GridTriangleDebugConnector<Real>::TransferData()
{
    cudaCheck(cudaGraphicsMapResources(1, &m_particle_buf));
    viewer::Particle *dev_particles;
    size_t buffer_size;
    cudaCheck(cudaGraphicsResourceGetMappedPointer((void **)&dev_particles, &buffer_size, m_particle_buf));
    unsigned int num_particle = (unsigned int)(buffer_size / sizeof(viewer::Particle));
    GridTriangleDebugConnectorKernel::transfer_particle_data<Real><<<CUDA_GRID_SIZE(num_particle), CUDA_BLOCK_SIZE>>>(dev_particles, m_dev_data, num_particle);
    cudaCheck(cudaGraphicsUnmapResources(1, &m_particle_buf));

    cudaCheck(cudaGraphicsMapResources(1, &m_line_seg_buf));
    viewer::LineSeg *dev_line_segs;
    cudaCheck(cudaGraphicsResourceGetMappedPointer((void **)&dev_line_segs, &buffer_size, m_line_seg_buf));
    num_particle = (unsigned int)(buffer_size / sizeof(viewer::LineSeg));
    GridTriangleDebugConnectorKernel::transfer_line_seg_data<Real><<<CUDA_GRID_SIZE(num_particle), CUDA_BLOCK_SIZE>>>(dev_line_segs, m_dev_data, num_particle);
    cudaCheck(cudaGraphicsUnmapResources(1, &m_line_seg_buf));
}

template class GridTriangleDebugConnector<float>;
template class GridTriangleDebugConnector<double>;

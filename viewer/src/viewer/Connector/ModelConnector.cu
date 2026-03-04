#include "ModelConnector.cuh"
#include <cuda_runtime.h>
#include <viewer/Framework/RenderObject.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

namespace viewer {

template <typename Real>
ModelConnector<Real>::ModelConnector(
    const std::vector<std::shared_ptr<RenderObject>>& objects,
    Real* dev_position,
    Real* dev_orientation,
    unsigned int num_bodies)
    : m_objects(objects)
    , m_dev_position(dev_position)
    , m_dev_orientation(dev_orientation)
    , m_num_bodies(num_bodies)
{
    m_host_positions.resize(num_bodies * 3);
    m_host_orientations.resize(num_bodies * 4);
}

template <typename Real>
void ModelConnector<Real>::TransferData()
{
    cudaMemcpy(m_host_positions.data(), m_dev_position, sizeof(Real) * m_num_bodies * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(m_host_orientations.data(), m_dev_orientation, sizeof(Real) * m_num_bodies * 4, cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < m_num_bodies; ++i)
    {
        glm::vec3 position(
            static_cast<float>(m_host_positions[i * 3 + 0]),
            static_cast<float>(m_host_positions[i * 3 + 1]),
            static_cast<float>(m_host_positions[i * 3 + 2]));

        glm::quat orientation(
            static_cast<float>(m_host_orientations[i * 4 + 0]),
            static_cast<float>(m_host_orientations[i * 4 + 1]),
            static_cast<float>(m_host_orientations[i * 4 + 2]),
            static_cast<float>(m_host_orientations[i * 4 + 3]));

        glm::mat4 model_mat = glm::translate(glm::mat4(1.0f), position) * glm::toMat4(orientation);

        if (m_objects[i])
        {
            m_objects[i]->m_model_mat = model_mat;
        }
    }
}

template class ModelConnector<float>;
template class ModelConnector<double>;

}  // namespace viewer

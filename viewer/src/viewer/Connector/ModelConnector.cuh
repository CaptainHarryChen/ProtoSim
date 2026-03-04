#pragma once
#include <vector>
#include <memory>
#include "Connector.cuh"
#include <glm/glm.hpp>

namespace viewer {
class RenderObject;

template <typename Real>
class ModelConnector : public Connector
{
public:
    ModelConnector(
        const std::vector<std::shared_ptr<RenderObject>>& objects,
        Real* dev_position,
        Real* dev_orientation,
        unsigned int num_bodies);
    virtual ~ModelConnector() = default;

    virtual void TransferData() override;

protected:
    std::vector<std::shared_ptr<RenderObject>> m_objects;
    Real* m_dev_position;
    Real* m_dev_orientation;
    unsigned int m_num_bodies;

    std::vector<Real> m_host_positions;
    std::vector<Real> m_host_orientations;
};
}  // namespace viewer

extern template class viewer::ModelConnector<float>;
extern template class viewer::ModelConnector<double>;

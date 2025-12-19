#pragma once
#include <memory>
#include "Connector.cuh"

struct cudaGraphicsResource;

namespace viewer {
class Mesh;

template <typename Real>
class MeshConnector : public Connector
{
public:
    MeshConnector(std::shared_ptr<Mesh> mesh, Real* dev_position);
    virtual ~MeshConnector();

    virtual void TransferData() override;

protected:
    std::shared_ptr<Mesh> m_mesh;
    Real*                 m_dev_position;

    cudaGraphicsResource* m_cuda_resource_buf;
};
}  // namespace viewer

extern template class viewer::MeshConnector<float>;
extern template class viewer::MeshConnector<double>;

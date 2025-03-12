#pragma once
#include <memory>
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <Connector/Connector.cuh>

class Mesh;

template <typename Real>
class MeshConnector : public Connector
{
public:
    MeshConnector(std::shared_ptr<Mesh> mesh, Real *dev_position);
    virtual ~MeshConnector();

    virtual void TransferData() override;

protected:
    std::shared_ptr<Mesh> m_mesh;
    Real *m_dev_position;

    cudaGraphicsResource_t m_cuda_resource_buf;
};

extern template class MeshConnector<float>;
extern template class MeshConnector<double>;

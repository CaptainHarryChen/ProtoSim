#pragma once
#include <memory>
#include <Connector/Connector.cuh>

class ParticleBatch;
template <typename Real>
struct PDMPMHybridSolverData;
struct cudaGraphicsResource;

template <typename Real>
class GridTriangleDebugConnector : public Connector
{
public:
    GridTriangleDebugConnector(std::shared_ptr<ParticleBatch> particles, PDMPMHybridSolverData<Real> *data, PDMPMHybridSolverData<Real> *dev_data);
    virtual ~GridTriangleDebugConnector();

    virtual void TransferData() override;

protected:
    std::shared_ptr<ParticleBatch> m_particles;
    PDMPMHybridSolverData<Real> *m_data;
    PDMPMHybridSolverData<Real> *m_dev_data;

    cudaGraphicsResource *m_cuda_resource_buf;
};

extern template class GridTriangleDebugConnector<float>;
extern template class GridTriangleDebugConnector<double>;

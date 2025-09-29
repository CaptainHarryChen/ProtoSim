#pragma once
#include <memory>
#include <Connector/Connector.cuh>

class ParticleBatch;
template <typename Real>
struct CPICSolverData;
struct cudaGraphicsResource;

template <typename Real>
class GridTriangleDebugConnector : public Connector
{
public:
    GridTriangleDebugConnector(std::shared_ptr<ParticleBatch> particles, CPICSolverData<Real> *data, CPICSolverData<Real> *dev_data);
    virtual ~GridTriangleDebugConnector();

    virtual void TransferData() override;

protected:
    std::shared_ptr<ParticleBatch> m_particles;
    CPICSolverData<Real> *m_data;
    CPICSolverData<Real> *m_dev_data;

    cudaGraphicsResource *m_cuda_resource_buf;
};

extern template class GridTriangleDebugConnector<float>;
extern template class GridTriangleDebugConnector<double>;

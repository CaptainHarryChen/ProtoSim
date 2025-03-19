#pragma once
#include <memory>
#include <Connector/Connector.cuh>

class LineSegment;
template <typename Real>
struct IQMPMSolverData;
struct cudaGraphicsResource;

template <typename Real>
class IQMPMDebugConnector : public Connector
{
public:
    IQMPMDebugConnector(std::shared_ptr<LineSegment> lines, IQMPMSolverData<Real> *data, unsigned int grid_id, Real scale);
    virtual ~IQMPMDebugConnector();

    virtual void TransferData() override;

protected:
    std::shared_ptr<LineSegment> m_lines;
    IQMPMSolverData<Real> *m_data;
    IQMPMSolverData<Real> *m_dev_data;
    unsigned int m_grid_id;
    Real m_scale;

    cudaGraphicsResource *m_cuda_resource_buf;
};

extern template class IQMPMDebugConnector<float>;
extern template class IQMPMDebugConnector<double>;

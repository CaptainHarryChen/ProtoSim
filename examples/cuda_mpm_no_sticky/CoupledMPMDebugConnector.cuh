#pragma once
#include <memory>
#include <Connector/Connector.cuh>

class LineSegment;
template <typename Real>
struct CoupledMPMSolverData;
struct cudaGraphicsResource;

template <typename Real>
class CoupledMPMDebugConnector : public Connector
{
public:
    CoupledMPMDebugConnector(std::shared_ptr<LineSegment> lines, CoupledMPMSolverData<Real> *data, unsigned int grid_id, Real scale);
    virtual ~CoupledMPMDebugConnector();

    virtual void TransferData() override;

protected:
    std::shared_ptr<LineSegment> m_lines;
    CoupledMPMSolverData<Real> *m_data;
    CoupledMPMSolverData<Real> *m_dev_data;
    unsigned int m_grid_id;
    Real m_scale;

    cudaGraphicsResource *m_cuda_resource_buf;
};

extern template class CoupledMPMDebugConnector<float>;
extern template class CoupledMPMDebugConnector<double>;

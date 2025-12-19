#pragma once
#include <memory>
#include <viewer/Connector/Connector.cuh>

template <typename Real>
struct CoupledMPMSolverData;
struct cudaGraphicsResource;

namespace viewer
{
    class LineSegment;
}

template <typename Real>
class CoupledMPMDebugConnector : public viewer::Connector
{
public:
    CoupledMPMDebugConnector(std::shared_ptr<viewer::LineSegment> lines, CoupledMPMSolverData<Real> *data, unsigned int grid_id, Real scale);
    virtual ~CoupledMPMDebugConnector();

    virtual void TransferData() override;

protected:
    std::shared_ptr<viewer::LineSegment> m_lines;
    CoupledMPMSolverData<Real> *m_data;
    CoupledMPMSolverData<Real> *m_dev_data;
    unsigned int m_grid_id;
    Real m_scale;

    cudaGraphicsResource *m_cuda_resource_buf;
};

extern template class CoupledMPMDebugConnector<float>;
extern template class CoupledMPMDebugConnector<double>;

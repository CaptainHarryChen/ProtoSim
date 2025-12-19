#pragma once
#include <memory>
#include <viewer/Connector/Connector.cuh>

template <typename Real>
struct MultiGridMPMSolverData;
struct cudaGraphicsResource;

namespace viewer
{
    class LineSegment;
}

template <typename Real>
class MultiGridMPMDebugConnector : public viewer::Connector
{
public:
    MultiGridMPMDebugConnector(std::shared_ptr<viewer::LineSegment> lines, MultiGridMPMSolverData<Real> *data, unsigned int grid_id, Real scale);
    virtual ~MultiGridMPMDebugConnector();

    virtual void TransferData() override;

protected:
    std::shared_ptr<viewer::LineSegment> m_lines;
    MultiGridMPMSolverData<Real> *m_data;
    MultiGridMPMSolverData<Real> *m_dev_data;
    unsigned int m_grid_id;
    Real m_scale;

    cudaGraphicsResource *m_cuda_resource_buf;
};

extern template class MultiGridMPMDebugConnector<float>;
extern template class MultiGridMPMDebugConnector<double>;

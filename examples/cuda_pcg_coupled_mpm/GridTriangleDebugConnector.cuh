#pragma once
#include <memory>
#include <Connector/Connector.cuh>

class ParticleBatch;
class LineSegment;
template <typename Real>
struct PCGCoupledMPMSolverData;
struct cudaGraphicsResource;

template <typename Real>
class GridTriangleDebugConnector : public Connector
{
public:
    GridTriangleDebugConnector(std::shared_ptr<ParticleBatch> particles, std::shared_ptr<LineSegment> line_segs, PCGCoupledMPMSolverData<Real> *data, PCGCoupledMPMSolverData<Real> *dev_data);
    virtual ~GridTriangleDebugConnector();

    virtual void TransferData() override;

protected:
    std::shared_ptr<ParticleBatch> m_particles;
    std::shared_ptr<LineSegment> m_line_segs;
    PCGCoupledMPMSolverData<Real> *m_data;
    PCGCoupledMPMSolverData<Real> *m_dev_data;

    cudaGraphicsResource *m_particle_buf;
    cudaGraphicsResource *m_line_seg_buf;
};

extern template class GridTriangleDebugConnector<float>;
extern template class GridTriangleDebugConnector<double>;

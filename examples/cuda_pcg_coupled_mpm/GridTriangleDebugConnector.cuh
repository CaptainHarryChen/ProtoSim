#pragma once
#include <memory>
#include <viewer/Connector/Connector.cuh>
#include <Solver/PCGCoupledMPMSolver.cuh>

namespace viewer
{
    class ParticleBatch;
    class LineSegment;
}

struct cudaGraphicsResource;

template <typename Real>
class GridTriangleDebugConnector : public viewer::Connector
{
public:
    GridTriangleDebugConnector(std::shared_ptr<viewer::ParticleBatch> particles, std::shared_ptr<viewer::LineSegment> line_segs, PCGCoupledMPMSolverData<Real> data);
    virtual ~GridTriangleDebugConnector();

    virtual void TransferData() override;

protected:
    std::shared_ptr<viewer::ParticleBatch> m_particles;
    std::shared_ptr<viewer::LineSegment> m_line_segs;
    PCGCoupledMPMSolverData<Real> m_data;

    cudaGraphicsResource *m_particle_buf;
    cudaGraphicsResource *m_line_seg_buf;
};

extern template class GridTriangleDebugConnector<float>;
extern template class GridTriangleDebugConnector<double>;

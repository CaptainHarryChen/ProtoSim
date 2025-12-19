#pragma once
#include <memory>
#include "Connector.cuh"

struct cudaGraphicsResource;

namespace viewer {
class ParticleBatch;

template <typename Real>
class ParticleConnector : public Connector
{
public:
    ParticleConnector(std::shared_ptr<ParticleBatch> particle_batch, Real* dev_position, Real* dev_color);
    virtual ~ParticleConnector();

    virtual void TransferData() override;

protected:
    std::shared_ptr<ParticleBatch> m_particle_batch;
    Real*                          m_dev_position;
    Real*                          m_dev_color;

    cudaGraphicsResource* m_cuda_resource_buf;
};
}  // namespace viewer

extern template class viewer::ParticleConnector<float>;
extern template class viewer::ParticleConnector<double>;

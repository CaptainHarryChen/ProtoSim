#pragma once
#include <memory>
#include <Connector/Connector.cuh>

class ParticleBatch;
struct cudaGraphicsResource;

template <typename Real>
class ParticleConnector : public Connector
{
public:
    ParticleConnector(std::shared_ptr<ParticleBatch> particle_batch, Real *dev_position, Real *dev_color);
    virtual ~ParticleConnector();

    virtual void TransferData() override;

protected:
    std::shared_ptr<ParticleBatch> m_particle_batch;
    Real *m_dev_position;
    Real *m_dev_color;

    cudaGraphicsResource *m_cuda_resource_buf;
};

extern template class ParticleConnector<float>;
extern template class ParticleConnector<double>;

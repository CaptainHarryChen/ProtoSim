#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <memory>
#include <viewer/Framework/RenderObject.h>

namespace viewer {

struct Particle
{
    glm::vec3 Position;
    glm::vec3 Color;
};

class ParticleBatch : public RenderObject
{
public:
    ParticleBatch(const std::vector<Particle>& particles);
    virtual ~ParticleBatch() = default;

    std::vector<Particle> m_particles;

    virtual void                UpdateParticles(const std::vector<Particle>& data);
    virtual void                DrawVAO() const override;
    virtual inline unsigned int GetVBO() const
    {
        return m_VBO;
    }

protected:
    template <typename Real>
    friend class ParticleConnector;

    unsigned int m_VAO;
    unsigned int m_VBO;
};

}  // namespace viewer

#include "ParticleBatch.h"
#include <glad/glad.h>

ParticleBatch::ParticleBatch(std::vector<Particle> particles) : m_particles(particles)
{
    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);
    glGenBuffers(1, &m_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, m_particles.size() * sizeof(Particle), &m_particles[0], GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void *)offsetof(Particle, Color));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
    glEnable(GL_PROGRAM_POINT_SIZE);
}

void ParticleBatch::DrawVAO() const
{
    glBindVertexArray(m_VAO);
    glDrawArrays(GL_POINTS, 0, (GLsizei)m_particles.size());
    glBindVertexArray(0);
}

void ParticleBatch::UpdateParticles(const std::vector<Particle> &data)
{
    glBindVertexArray(m_VAO);
    m_particles = data;
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, m_particles.size() * sizeof(Particle), &m_particles[0]);
    glBindVertexArray(0);
}

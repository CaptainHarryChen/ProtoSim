#include "SingleCapsule.h"
#include <glad/glad.h>

namespace viewer
{

SingleCapsule::SingleCapsule(glm::vec3 color) : m_color(color)
{
    struct PointData
    {
        float position[3];
        float color[3];
    };

    PointData point = {
        {0.0f, 0.0f, 0.0f},
        {color.x, color.y, color.z}
    };

    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);
    glGenBuffers(1, &m_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(PointData), &point, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, (void *)(sizeof(float) * 3));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

void SingleCapsule::DrawVAO() const
{
    glBindVertexArray(m_VAO);
    glDrawArrays(GL_POINTS, 0, 1);
    glBindVertexArray(0);
}

glm::mat4 SingleCapsule::GetModelMatrix() const
{
    return m_model_mat;
}

void SingleCapsule::SetColor(glm::vec3 color)
{
    m_color = color;

    struct PointData
    {
        float position[3];
        float color[3];
    };

    PointData point = {
        {0.0f, 0.0f, 0.0f},
        {color.x, color.y, color.z}
    };

    glBindVertexArray(m_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(PointData), &point);
    glBindVertexArray(0);
}

}

#include "LineSegment.h"
#include <glad/glad.h>
#include <Render/Shader.h>

LineSegment::LineSegment(std::vector<glm::vec3> vertices, std::vector<std::pair<int, int>> connectivity)
{
    for (int i = 0; i < connectivity.size(); ++i)
        m_line_segs.push_back({vertices[connectivity[i].first], vertices[connectivity[i].second]});
    
    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);
    glGenBuffers(1, &m_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, m_line_segs.size() * sizeof(LineSeg), m_line_segs.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
    glLineWidth(2.0);
}

void LineSegment::DrawVAO() const
{
    glBindVertexArray(m_VAO);
    glDrawArrays(GL_LINES, 0, 2 * (GLsizei)m_line_segs.size());
    glBindVertexArray(0);
}

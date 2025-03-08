#pragma once
#include <vector>
#include <memory>
#include <Render/RenderObject.h>

struct Vertex
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 tex_coords;
};

class Mesh : public RenderObject
{
public:
    Mesh(const std::vector<Vertex> &vertices, const std::vector<unsigned int> &indices);
    virtual ~Mesh() = default;

    std::vector<Vertex> m_vertices;
    std::vector<unsigned int> m_indices;

    virtual void UpdateVertices(const std::vector<Vertex> &data);
    virtual void DrawVAO() const override;
protected:
    unsigned int m_VAO;
    unsigned int m_VBO, EBO;
};

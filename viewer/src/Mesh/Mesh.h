#pragma once
#include <vector>
#include <memory>
#include <Render/RenderObject.h>

struct Vertex
{
	glm::vec3 Position;
	glm::vec3 Normal;
	glm::vec2 TexCoords;
};

class Mesh : public RenderObject
{
public:
    Mesh(const std::vector<Vertex> &vertices, const std::vector<unsigned int> &indices);
    virtual ~Mesh() = default;

    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    virtual void UpdateVertices(const std::vector<Vertex> &data);
    virtual void DrawVAO() const;
protected:
    unsigned int VAO;
    unsigned int VBO, EBO;
};

#pragma once
#include <vector>
#include <memory>
#include <viewer/Framework/RenderObject.h>

namespace viewer {

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 tex_coords;
};

class Mesh : public RenderObject
{
public:
    Mesh(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices, bool enable_shadow = true);
    virtual ~Mesh() = default;

    bool                      m_enable_shadow = true;
    std::vector<Vertex>       m_vertices;
    std::vector<unsigned int> m_indices;

    virtual void      UpdateVertices(const std::vector<Vertex>& data);
    virtual void      DrawVAO() const override;
    virtual glm::mat4 GetModelMatrix() const override;

protected:
    template <typename Real>
    friend class MeshConnector;

    unsigned int m_VAO;
    unsigned int m_VBO;
    unsigned int m_EBO;
};

}  // namespace viewer

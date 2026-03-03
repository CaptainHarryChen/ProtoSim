#pragma once

#include <glm/glm.hpp>
#include <viewer/Framework/RenderObject.h>

namespace viewer
{

class SingleCapsule : public RenderObject
{
public:
    SingleCapsule(glm::vec3 color = glm::vec3(1.0f));
    virtual ~SingleCapsule() = default;

    glm::mat4 m_model_mat = glm::mat4(1.0f);

    virtual void DrawVAO() const override;
    virtual glm::mat4 GetModelMatrix() const override;

    void SetColor(glm::vec3 color);

protected:
    unsigned int m_VAO;
    unsigned int m_VBO;
    glm::vec3 m_color;
};

}

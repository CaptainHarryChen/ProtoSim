#pragma once
#include <Mesh/MeshBase.h>

class ShadowMapping;
class RenderObject;

class CubeLight : public MeshBase // render light cube + render shadow map
{
public:
    CubeLight(glm::vec3 position, glm::vec3 color, bool isCubic = true);

    glm::vec3 getPos();
    glm::vec3 getColor();
    bool IsOn();
    void setIsOn(bool on);

    unsigned int getDepthMap();
    void generateShadowMap(std::vector<std::shared_ptr<RenderObject>> &objects);
    glm::mat4 getlightSpaceMatrix();
    float getFarPlane();

    template <typename T>
    void setPos(T p)
    {
        lightPos = {p[0], p[1], p[2]};
    }

    template <typename T>
    void setColor(T c)
    {
        lightColor = {c[0], c[1], c[2]};
    }

private:
    static std::vector<Vertex> __CUBE_VERTICES;
    static std::vector<unsigned int> __CUBE_INDICES;

    std::shared_ptr<ShadowMapping> shadowMapping;
    glm::vec3 lightPos;
    glm::vec3 lightColor;
    bool isOn = true;
};

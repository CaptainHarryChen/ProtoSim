#pragma once
#include <Mesh/MeshBase.h>

class ShadowMapping;

class Light : public MeshBase // render light cube + render shadow map
{
public:
    Light(glm::vec3 position, glm::vec3 color, std::vector<std::shared_ptr<SurfaceMesh>> meshes, bool isCubic = true);

    std::shared_ptr<ShadowMapping> shadowMapping;
    std::vector<std::shared_ptr<SurfaceMesh>> shadowMeshes;

    glm::vec3 getPos();
    unsigned int getDepthMap();
    void generateShadowMap();
    glm::mat4 getlightSpaceMatrix();
    glm::vec3 getColor();

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

    glm::vec3 lightPos;
    glm::vec3 lightColor;
};

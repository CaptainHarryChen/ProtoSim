#pragma once
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <Mesh/MeshBase.h>

class Light;

class CubeLight : public MeshBase
{
public:
    CubeLight(glm::vec3 position, glm::vec3 color);
    virtual ~CubeLight() = default;

    std::shared_ptr<Light> getLight();

protected:
    std::shared_ptr<Light> light;

private:
    static std::vector<Vertex> __CUBE_VERTICES;
    static std::vector<unsigned int> __CUBE_INDICES;
};

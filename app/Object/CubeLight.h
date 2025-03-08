#pragma once
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <Object/Object.h>
#include <Mesh/Mesh.h>

class Light;

class CubeLight : public Object
{
public:
    CubeLight(glm::vec3 position, glm::vec3 color);
    virtual ~CubeLight() = default;

    std::shared_ptr<Mesh> GetMesh();
    std::shared_ptr<Light> GetLight();
    void SetPosition(const glm::vec3 &position);
    void SetColor(const glm::vec3 &color);
    void SetLightOn(bool isOn);

protected:
    std::shared_ptr<Mesh> m_mesh;
    std::shared_ptr<Light> m_light;

private:
    static std::vector<Vertex> __CUBE_VERTICES;
    static std::vector<unsigned int> __CUBE_INDICES;
};

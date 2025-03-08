#include "CubeLight.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <Mesh/Mesh.h>
#include <Mesh/SolidColorRenderer.h>
#include <Render/Light.h>

CubeLight::CubeLight(glm::vec3 position, glm::vec3 color)
{
    light = std::make_shared<Light>(position, color, 0.2f, 1000.f);
    mesh = std::make_shared<Mesh>(__CUBE_VERTICES, __CUBE_INDICES);
    mesh->model = glm::translate(mesh->model, position);
    mesh->model = glm::scale(mesh->model, glm::vec3(0.2f));
    mesh->AddRenderer(std::make_shared<SolidColorRenderer>(color));
}

std::shared_ptr<Mesh> CubeLight::GetMesh()
{
    return mesh;
}

std::shared_ptr<Light> CubeLight::GetLight()
{
    return light;
}

void CubeLight::SetPosition(const glm::vec3 &position)
{
    light->lightPos = position;
    mesh->model = glm::translate(glm::mat4(1.0f), position);
    mesh->model = glm::scale(mesh->model, glm::vec3(0.2f));
}

void CubeLight::SetColor(const glm::vec3 &color)
{
    light->lightColor = color;
    for (auto &renderer : mesh->renderers)
    {
        auto solid_color_renderer = std::dynamic_pointer_cast<SolidColorRenderer>(renderer);
        if (solid_color_renderer)
            solid_color_renderer->SetColor(color);
    }
}

void CubeLight::SetLightOn(bool isOn)
{
    light->isOn = isOn;
    mesh->m_enable = isOn;
}

std::vector<Vertex> CubeLight::__CUBE_VERTICES = {
    // positions           // normals              // texture coords
    {{-0.5f, 0.0f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
    {{0.5f, 1.0f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
    {{0.5f, 0.0f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},

    {{0.5f, 1.0f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
    {{-0.5f, 0.0f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
    {{-0.5f, 1.0f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

    {{-0.5f, 0.0f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.0f, 0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
    {{0.5f, 1.0f, 0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},

    {{0.5f, 1.0f, 0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
    {{-0.5f, 1.0f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.0f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},

    {{-0.5f, 1.0f, 0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{-0.5f, 1.0f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
    {{-0.5f, 0.0f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

    {{-0.5f, 0.0f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.0f, 0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
    {{-0.5f, 1.0f, 0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},

    {{0.5f, 1.0f, 0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, 0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
    {{0.5f, 1.0f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},

    {{0.5f, 0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
    {{0.5f, 1.0f, 0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, 0.0f, 0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},

    {{-0.5f, 0.0f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
    {{0.5f, 0.0f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
    {{0.5f, 0.0f, 0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},

    {{0.5f, 0.0f, 0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
    {{-0.5f, 0.0f, 0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
    {{-0.5f, 0.0f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

    {{-0.5f, 1.0f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
    {{0.5f, 1.0f, 0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, 1.0f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},

    {{0.5f, 1.0f, 0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
    {{-0.5f, 1.0f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
    {{-0.5f, 1.0f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}}};

std::vector<unsigned int> CubeLight::__CUBE_INDICES = {
    0, 1, 2,
    3, 4, 5,
    6, 7, 8,
    9, 10, 11,
    12, 13, 14,
    15, 16, 17,
    18, 19, 20,
    21, 22, 23,
    24, 25, 26,
    27, 28, 29,
    30, 31, 32,
    33, 34, 35};

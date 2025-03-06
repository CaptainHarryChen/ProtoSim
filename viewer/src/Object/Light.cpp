#include "Light.h"
#include <Geometry/ShadowMapping.h>

Light::Light(glm::vec3 position, glm::vec3 color, std::vector<std::shared_ptr<SurfaceMesh>> meshes, bool isCubic)
{
    // std::cout << isCubic << std::endl;
    lightPos = position;
    lightColor = color;
    shadowMeshes = meshes;
    shadowMapping = std::make_shared<ShadowMapping>(isCubic);
    mesh = std::make_shared<SurfaceMesh>(__CUBE_VERTICES, __CUBE_INDICES);
}
glm::vec3 Light::getPos()
{
    return lightPos;
}

unsigned int Light::getDepthMap()
{
    return shadowMapping->getDepthMap();
}

void Light::generateShadowMap()
{
    // std::cout << lightPos.x << " " << lightPos.y <<  " " << lightPos.z << std::endl;
    // std::cout << shadowMeshes.size() << std::endl;
    shadowMapping->Draw(lightPos, shadowMeshes);
}

glm::mat4 Light::getlightSpaceMatrix()
{
    return shadowMapping->getlightSpaceMatrix();
}

glm::vec3 Light::getColor()
{
    return lightColor;
}

std::vector<Vertex> Light::__CUBE_VERTICES = {
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

std::vector<unsigned int> Light::__CUBE_INDICES = {
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

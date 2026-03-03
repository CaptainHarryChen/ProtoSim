#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <viewer/RenderObject/Mesh.h>

namespace viewer
{
    namespace PrimitiveGenerator
    {
        std::shared_ptr<Mesh> GenerateSphere(float radius, unsigned int segments = 32, unsigned int rings = 16);
        std::shared_ptr<Mesh> GenerateBox(glm::vec3 half_extents);
        std::shared_ptr<Mesh> GenerateCapsule(float radius, float half_height, unsigned int segments = 16, unsigned int rings = 8);
    }
}

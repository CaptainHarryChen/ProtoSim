#include "PrimitiveGenerator.h"
#include <cmath>
#include <glm/gtc/constants.hpp>

namespace viewer
{

    std::shared_ptr<Mesh> PrimitiveGenerator::GenerateSphere(float radius, unsigned int segments, unsigned int rings)
    {
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;

        for (unsigned int ring = 0; ring <= rings; ++ring)
        {
            float theta = ring * glm::pi<float>() / rings;
            float sin_theta = std::sin(theta);
            float cos_theta = std::cos(theta);

            for (unsigned int seg = 0; seg <= segments; ++seg)
            {
                float phi = seg * 2.0f * glm::pi<float>() / segments;
                float sin_phi = std::sin(phi);
                float cos_phi = std::cos(phi);

                glm::vec3 normal(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
                glm::vec3 position = normal * radius;
                glm::vec2 tex_coords(static_cast<float>(seg) / segments, static_cast<float>(ring) / rings);

                vertices.push_back({position, normal, tex_coords});
            }
        }

        for (unsigned int ring = 0; ring < rings; ++ring)
        {
            for (unsigned int seg = 0; seg < segments; ++seg)
            {
                unsigned int current = ring * (segments + 1) + seg;
                unsigned int next = current + segments + 1;

                indices.push_back(current);
                indices.push_back(next);
                indices.push_back(current + 1);

                indices.push_back(current + 1);
                indices.push_back(next);
                indices.push_back(next + 1);
            }
        }

        return std::make_shared<Mesh>(vertices, indices);
    }

    std::shared_ptr<Mesh> PrimitiveGenerator::GenerateBox(glm::vec3 half_extents)
    {
        float hx = half_extents.x;
        float hy = half_extents.y;
        float hz = half_extents.z;

        std::vector<Vertex> vertices = {
            {{-hx, -hy, -hz}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
            {{hx, hy, -hz}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
            {{hx, -hy, -hz}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
            {{hx, hy, -hz}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
            {{-hx, -hy, -hz}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
            {{-hx, hy, -hz}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

            {{-hx, -hy, hz}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
            {{hx, -hy, hz}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
            {{hx, hy, hz}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
            {{hx, hy, hz}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
            {{-hx, hy, hz}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
            {{-hx, -hy, hz}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},

            {{-hx, hy, hz}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
            {{-hx, hy, -hz}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
            {{-hx, -hy, -hz}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
            {{-hx, -hy, -hz}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
            {{-hx, -hy, hz}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
            {{-hx, hy, hz}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},

            {{hx, hy, hz}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
            {{hx, -hy, -hz}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
            {{hx, hy, -hz}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
            {{hx, -hy, -hz}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
            {{hx, hy, hz}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
            {{hx, -hy, hz}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},

            {{-hx, -hy, -hz}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
            {{hx, -hy, -hz}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
            {{hx, -hy, hz}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
            {{hx, -hy, hz}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
            {{-hx, -hy, hz}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
            {{-hx, -hy, -hz}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

            {{-hx, hy, -hz}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
            {{hx, hy, hz}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
            {{hx, hy, -hz}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
            {{hx, hy, hz}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
            {{-hx, hy, -hz}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
            {{-hx, hy, hz}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        };

        std::vector<unsigned int> indices;
        for (unsigned int i = 0; i < 36; ++i)
        {
            indices.push_back(i);
        }

        return std::make_shared<Mesh>(vertices, indices);
    }

    std::shared_ptr<Mesh> PrimitiveGenerator::GenerateCapsule(float radius, float half_height, unsigned int segments, unsigned int rings)
    {
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;

        for (unsigned int ring = 0; ring <= rings; ++ring)
        {
            float theta = ring * glm::pi<float>() / (2 * rings);
            float sin_theta = std::sin(theta);
            float cos_theta = std::cos(theta);

            for (unsigned int seg = 0; seg <= segments; ++seg)
            {
                float phi = seg * 2.0f * glm::pi<float>() / segments;
                float sin_phi = std::sin(phi);
                float cos_phi = std::cos(phi);

                glm::vec3 normal(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
                float y_offset = half_height;
                glm::vec3 position = normal * radius;
                position.y += y_offset;

                glm::vec2 tex_coords(static_cast<float>(seg) / segments, static_cast<float>(ring) / (2 * rings));

                vertices.push_back({position, normal, tex_coords});
            }
        }

        for (unsigned int ring = 0; ring <= rings; ++ring)
        {
            float theta = glm::pi<float>() / 2 + ring * glm::pi<float>() / (2 * rings);
            float sin_theta = std::sin(theta);
            float cos_theta = std::cos(theta);

            for (unsigned int seg = 0; seg <= segments; ++seg)
            {
                float phi = seg * 2.0f * glm::pi<float>() / segments;
                float sin_phi = std::sin(phi);
                float cos_phi = std::cos(phi);

                glm::vec3 normal(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
                float y_offset = -half_height;
                glm::vec3 position = normal * radius;
                position.y += y_offset;

                glm::vec2 tex_coords(static_cast<float>(seg) / segments, static_cast<float>(rings + ring) / (2 * rings));

                vertices.push_back({position, normal, tex_coords});
            }
        }

        unsigned int cylinder_start = static_cast<unsigned int>(vertices.size());
        for (unsigned int seg = 0; seg <= segments; ++seg)
        {
            float phi = seg * 2.0f * glm::pi<float>() / segments;
            float sin_phi = std::sin(phi);
            float cos_phi = std::cos(phi);

            glm::vec3 normal(cos_phi, 0.0f, sin_phi);

            vertices.push_back({glm::vec3(radius * cos_phi, half_height, radius * sin_phi), normal, {static_cast<float>(seg) / segments, 0.5f}});
            vertices.push_back({glm::vec3(radius * cos_phi, -half_height, radius * sin_phi), normal, {static_cast<float>(seg) / segments, 0.5f}});
        }

        for (unsigned int ring = 0; ring < rings; ++ring)
        {
            for (unsigned int seg = 0; seg < segments; ++seg)
            {
                unsigned int current = ring * (segments + 1) + seg;
                unsigned int next = current + segments + 1;

                indices.push_back(current);
                indices.push_back(next);
                indices.push_back(current + 1);

                indices.push_back(current + 1);
                indices.push_back(next);
                indices.push_back(next + 1);
            }
        }

        unsigned int bottom_hemisphere_start = (rings + 1) * (segments + 1);
        for (unsigned int ring = 0; ring < rings; ++ring)
        {
            for (unsigned int seg = 0; seg < segments; ++seg)
            {
                unsigned int current = bottom_hemisphere_start + ring * (segments + 1) + seg;
                unsigned int next = current + segments + 1;

                indices.push_back(current);
                indices.push_back(next);
                indices.push_back(current + 1);

                indices.push_back(current + 1);
                indices.push_back(next);
                indices.push_back(next + 1);
            }
        }

        for (unsigned int seg = 0; seg < segments; ++seg)
        {
            unsigned int top_ring_last = rings * (segments + 1) + seg;
            unsigned int top_cylinder = cylinder_start + seg * 2;

            indices.push_back(top_ring_last);
            indices.push_back(top_ring_last + 1);
            indices.push_back(top_cylinder);

            unsigned int bottom_ring_first = bottom_hemisphere_start + seg;
            unsigned int bottom_cylinder = cylinder_start + seg * 2 + 1;

            indices.push_back(bottom_cylinder);
            indices.push_back(bottom_ring_first + 1);
            indices.push_back(bottom_ring_first);
        }

        for (unsigned int seg = 0; seg < segments; ++seg)
        {
            unsigned int current = cylinder_start + seg * 2;
            unsigned int next = cylinder_start + (seg + 1) * 2;

            indices.push_back(current);
            indices.push_back(next);
            indices.push_back(current + 1);

            indices.push_back(current + 1);
            indices.push_back(next);
            indices.push_back(next + 1);
        }

        return std::make_shared<Mesh>(vertices, indices);
    }

}

#include "Floor.h"
#include <string>
#include <Mesh/Mesh.h>
#include <Mesh/TextureRenderer.h>
#include <viewer_config.h>

Floor::Floor(float scale)
{
    std::vector<Vertex> __vertices = __PLANE_VERTICES;
    for (int i = 0; i < __vertices.size(); ++i)
    {
        __vertices[i].position *= scale;
        __vertices[i].tex_coords *= scale;
    }
    std::string albedo = std::string(VIEWER_DIR) + "/data/floor/albedo.png";
    std::string metallic = std::string(VIEWER_DIR) + "/data/floor/metallic.png";
    std::string normal = std::string(VIEWER_DIR) + "/data/floor/normal.png";
    std::string roughness = std::string(VIEWER_DIR) + "/data/floor/roughness.png";
    std::string ao = std::string(VIEWER_DIR) + "/data/floor/ao.png";
    std::vector<std::string> textures = {albedo, normal, metallic, roughness, ao};
    m_mesh = std::make_shared<Mesh>(__vertices, __PLANE_INDICES);
    m_mesh->AddRenderer(std::make_shared<TextureRenderer>(textures));
}

std::shared_ptr<Mesh> Floor::GetMesh()
{
    return m_mesh;
}

std::vector<Vertex> Floor::__PLANE_VERTICES = {{{-5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
                                        {{5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {5.0f, 0.0f}},
                                        {{-5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 5.0f}},
                                        {{5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {5.0f, 5.0f}}};

std::vector<unsigned int> Floor::__PLANE_INDICES = {0, 1, 2,
                                             1, 3, 2};

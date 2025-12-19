#include "MeshLoader.h"
#include <memory>
#include <filesystem>
#include <iostream>
#include <sys/stat.h>
#include <glm/gtc/quaternion.hpp>
#include <tiny_obj_loader.h>
#include <viewer/RenderObject/Mesh.h>
#include <viewer/Renderer/PbrRenderer.h>
#include <viewer/Renderer/TextureRenderer.h>

namespace viewer {

namespace {
glm::vec3 rotateVecQuat(glm::quat q, glm::vec3 v)
{
    glm::vec3 u(q.x, q.y, q.z);
    float     s = q.w;
    return 2.0f * glm::dot(u, v) * u + (s * s - glm::dot(u, u)) * v + 2.0f * s * glm::cross(u, v);
}
}  // namespace

namespace MeshLoader {

std::shared_ptr<Mesh> LoadMesh(std::string inputfile, float scale, glm::vec3 translate,
                               glm::vec3              rotate,  // pitch, yaw, roll
                               std::vector<glm::vec3> material)
{
    tinyobj::ObjReaderConfig  reader_config;
    tinyobj::ObjReader        reader;
    std::vector<Vertex>       vertices;
    std::vector<unsigned int> indices;

    if (!reader.ParseFromFile(inputfile, reader_config))
    {
        std::cerr << "illegal .obj file: " << inputfile << std::endl;
        exit(1);
    }

    auto& attrib    = reader.GetAttrib();
    auto& shapes    = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    int global_v = 0;
    int global_f = 0;
    // Loop over shapes

    size_t    nVerts = attrib.vertices.size() / 3;
    glm::quat q(rotate);
    for (size_t v = 0; v < nVerts; ++v)
    {
        glm::vec3 verts(attrib.vertices[3 * v], attrib.vertices[3 * v + 1], attrib.vertices[3 * v + 2]);
        verts = rotateVecQuat(q, verts);

        vertices.push_back(
            Vertex(
                { scale * verts + translate,
                  glm::vec3(0.),
                  glm::vec2(0.) }));
    }

    int numTris = 0;
    for (size_t s = 0; s < shapes.size(); s++)
        numTris += ( int )shapes[s].mesh.num_face_vertices.size();
    indices.resize(numTris * 3);

    int f_cnt = 0;
    for (size_t s = 0; s < shapes.size(); s++)
    {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
        {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            // assert(fv == 3);
            // if (fv != 3) std::cout << fv << std::endl;
            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++)
            {
                // access to vertex
                tinyobj::index_t idx    = shapes[s].mesh.indices[index_offset + v];
                indices[fv * f_cnt + v] = idx.vertex_index;
            }
            ++f_cnt;
            index_offset += fv;
        }
    }

    auto mesh = std::make_shared<Mesh>(vertices, indices);
    mesh->AddRenderer(std::make_shared<PbrRenderer>(material));
    return mesh;
}

std::shared_ptr<Mesh> LoadMesh(std::string inputfile, float scale, glm::vec3 translate,
                               glm::vec3   rotate,  // pitch, yaw, roll
                               std::string albedo,
                               std::string metallic,
                               std::string normal,
                               std::string roughness,
                               std::string ao)
{
    struct stat buffer;
    if (stat(inputfile.c_str(), &buffer) != 0)
    {
        std::cerr << ".obj file missed " << inputfile << std::endl;
    }

    std::vector<Vertex>       vertices;
    std::vector<unsigned int> indices;

    glm::quat q(rotate);
    // glm::mat4 rot = glm::toMat4(q);
    // glm::mat4 trans = glm::translate(glm::mat4(1.0f), translate);
    // glm::mat4 transform = scale * rot * trans;

    tinyobj::attrib_t                attributes;
    std::vector<tinyobj::material_t> materials;
    std::vector<tinyobj::shape_t>    shapes;

    std::string warning, error;

    if (!tinyobj::LoadObj(&attributes, &shapes, &materials, &warning, &error, inputfile.c_str()))
    {
        std::cout << warning << error << "\n";
    }

    unsigned int itTris = 0;
    for (const auto& shape : shapes)
    {
        for (const auto& index : shape.mesh.indices)
        {
            glm::vec3 __vertex(
                attributes.vertices[3 * index.vertex_index],
                attributes.vertices[3 * index.vertex_index + 1],
                attributes.vertices[3 * index.vertex_index + 2]);
            // __vertex = glm::vec3(transform * glm::vec4(__vertex, 1.));
            __vertex = rotateVecQuat(q, __vertex);

            glm::vec2 __texCoord(
                attributes.texcoords[2 * index.texcoord_index],
                attributes.texcoords[2 * index.texcoord_index + 1]);

            vertices.push_back({ scale * __vertex + translate, glm::vec3(0.), __texCoord });
            indices.push_back(3 * itTris);
            indices.push_back(3 * itTris + 1);
            indices.push_back(3 * itTris + 2);
            ++itTris;
        }
    }
    std::vector<std::string> textures = { albedo, normal, metallic, roughness, ao };
    auto                     mesh     = std::make_shared<Mesh>(vertices, indices);
    mesh->AddRenderer(std::make_shared<TextureRenderer>(textures));
    return mesh;
}
}  // namespace MeshLoader

}  // namespace viewer

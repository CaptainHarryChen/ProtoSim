#include "TetrahedronLoader.h"
#include <fstream>
#include <cassert>
#include <map>
#include <vector>
#include <algorithm>
#include <glm/gtc/quaternion.hpp>

namespace TetrahedronLoader
{
    static glm::vec3 rotateVecQuat(glm::quat q, glm::vec3 v)
    {
        glm::vec3 u(q.x, q.y, q.z);
        float s = q.w;
        return 2.0f * glm::dot(u, v) * u + (s * s - glm::dot(u, u)) * v + 2.0f * s * glm::cross(u, v);
    }

    void LoadTetrahedron(std::string filename,
                         float scale, glm::vec3 translate, glm::vec3 rotate,
                         std::vector<float> &vertices, std::vector<unsigned int> &surface_triangles, std::vector<unsigned int> &tetrahedras)
    {
        // 0. .node file
        std::fstream nodeFS(filename + ".node");
        assert(nodeFS.is_open() && "tetgen_loader::ERROR .node file not found.");
        // Node count, 3 dim, no attribute, no boundary marker
        unsigned int nNodes, nDims, nNodeAttribs, nMarkers;
        nodeFS >> nNodes >> nDims >> nNodeAttribs >> nMarkers;
        bool nodeStartAtZero = false;
        vertices.resize(nNodes * 3);
        for (unsigned int i = 0; i < nNodes; ++i)
        {
            unsigned int _;
            glm::vec3 vert;
            nodeFS >> _ >> vert.x >> vert.y >> vert.z;
            vert = rotateVecQuat(glm::quat(rotate), vert);
            vert = scale * vert + translate;
            vertices[3 * i] = vert.x;
            vertices[3 * i + 1] = vert.y;
            vertices[3 * i + 2] = vert.z;
            if (i == 0 && _ == 0)
                nodeStartAtZero = true;
            for (unsigned int j = 0; j < nNodeAttribs + nMarkers; ++j)
            {
                nodeFS >> _;
            }
        }

        //  1. .ele file
        std::fstream eleFS(filename + ".ele");
        assert(eleFS.is_open() && "tetgen_loader::ERROR .ele file not found.");
        // <# of tetrahedra> <nodes per tetrahedron> <# of attributes>
        unsigned int nTets, nNodesPerTet, nEleAttribs;
        eleFS >> nTets >> nNodesPerTet >> nEleAttribs;
        tetrahedras.resize(nTets * 4);
        for (unsigned int i = 0; i < nTets; ++i)
        {
            unsigned int _;
            eleFS >> _ >> tetrahedras[4 * i] >> tetrahedras[4 * i + 1] >> tetrahedras[4 * i + 2] >> tetrahedras[4 * i + 3];
            if (nodeStartAtZero == false)
            {
                tetrahedras[4 * i] -= 1;
                tetrahedras[4 * i + 1] -= 1;
                tetrahedras[4 * i + 2] -= 1;
                tetrahedras[4 * i + 3] -= 1;
            }
            for (unsigned int j = 0; j < nEleAttribs; ++j)
            {
                eleFS >> _;
            }
        }

        // 2. generate surface triangles
        std::map<std::vector<unsigned int>, unsigned int> triangles;
        for (unsigned int i = 0; i < nTets; ++i)
        {
            for (unsigned int j = 0; j < 4; ++j)
            {
                std::vector<unsigned int> tri = {tetrahedras[4 * i + j], tetrahedras[4 * i + (j + 1) % 4], tetrahedras[4 * i + (j + 2) % 4]};
                std::sort(tri.begin(), tri.end());
                if (triangles.find(tri) == triangles.end())
                    triangles[tri] = tetrahedras[4 * i + (j + 3) % 4];
                else
                    triangles[tri] = -1;
            }
        }
        for (const auto &iter : triangles)
        {
            if (iter.second == -1)
                continue;
            unsigned int id_a = iter.first[0];
            unsigned int id_b = iter.first[1];
            unsigned int id_c = iter.first[2];
            unsigned int id_d = iter.second;
            glm::vec3 a = glm::vec3(vertices[id_a * 3], vertices[id_a * 3 + 1], vertices[id_a * 3 + 2]);
            glm::vec3 b = glm::vec3(vertices[id_b * 3], vertices[id_b * 3 + 1], vertices[id_b * 3 + 2]);
            glm::vec3 c = glm::vec3(vertices[id_c * 3], vertices[id_c * 3 + 1], vertices[id_c * 3 + 2]);
            glm::vec3 d = glm::vec3(vertices[id_d * 3], vertices[id_d * 3 + 1], vertices[id_d * 3 + 2]);
            if (glm::dot(glm::cross(b - a, c - a), d - a) > 0.0f)
                std::swap(id_b, id_c);
            surface_triangles.push_back(id_a);
            surface_triangles.push_back(id_b);
            surface_triangles.push_back(id_c);
        }
    }

    void LoadTetrahedronWithPLYSample(std::string filename,
                                      float scale, glm::vec3 translate, glm::vec3 rotate,
                                      std::vector<float> &vertices, std::vector<unsigned int> &surface_triangles, std::vector<unsigned int> &tetrahedras,
                                      std::vector<unsigned int> &sample_tri_idx, std::vector<float> &sample_barycentric_weights)
    {
        // 0. .node file
        std::fstream nodeFS(filename + ".node");
        assert(nodeFS.is_open() && "tetgen_loader::ERROR .node file not found.");
        // Node count, 3 dim, no attribute, no boundary marker
        unsigned int nNodes, nDims, nNodeAttribs, nMarkers;
        nodeFS >> nNodes >> nDims >> nNodeAttribs >> nMarkers;
        bool nodeStartAtZero = false;
        vertices.resize(nNodes * 3);
        for (unsigned int i = 0; i < nNodes; ++i)
        {
            unsigned int _;
            glm::vec3 vert;
            nodeFS >> _ >> vert.x >> vert.y >> vert.z;
            vert = rotateVecQuat(glm::quat(rotate), vert);
            vert = scale * vert + translate;
            vertices[3 * i] = vert.x;
            vertices[3 * i + 1] = vert.y;
            vertices[3 * i + 2] = vert.z;
            if (i == 0 && _ == 0)
                nodeStartAtZero = true;
            for (unsigned int j = 0; j < nNodeAttribs + nMarkers; ++j)
            {
                nodeFS >> _;
            }
        }

        //  1. .ele file
        std::fstream eleFS(filename + ".ele");
        assert(eleFS.is_open() && "tetgen_loader::ERROR .ele file not found.");
        // <# of tetrahedra> <nodes per tetrahedron> <# of attributes>
        unsigned int nTets, nNodesPerTet, nEleAttribs;
        eleFS >> nTets >> nNodesPerTet >> nEleAttribs;
        tetrahedras.resize(nTets * 4);
        for (unsigned int i = 0; i < nTets; ++i)
        {
            unsigned int _;
            eleFS >> _ >> tetrahedras[4 * i] >> tetrahedras[4 * i + 1] >> tetrahedras[4 * i + 2] >> tetrahedras[4 * i + 3];
            if (nodeStartAtZero == false)
            {
                tetrahedras[4 * i] -= 1;
                tetrahedras[4 * i + 1] -= 1;
                tetrahedras[4 * i + 2] -= 1;
                tetrahedras[4 * i + 3] -= 1;
            }
            for (unsigned int j = 0; j < nEleAttribs; ++j)
            {
                eleFS >> _;
            }
        }

        // 2. .face file
        std::fstream faceFS(filename + ".face");
        assert(faceFS.is_open() && "tetgen_loader::ERROR .face file not found.");
        // <# of faces> <# of attributes>
        unsigned int nFaces, nFaceAttributes;
        faceFS >> nFaces >> nFaceAttributes;
        surface_triangles.resize(nFaces * 3);
        for (unsigned int i = 0; i < nFaces; ++i)
        {
            unsigned int _;
            faceFS >> _ >> surface_triangles[3 * i] >> surface_triangles[3 * i + 2] >> surface_triangles[3 * i + 1];
            if (nodeStartAtZero == false)
            {
                surface_triangles[3 * i] -= 1;
                surface_triangles[3 * i + 1] -= 1;
                surface_triangles[3 * i + 2] -= 1;
            }
            for (unsigned int j = 0; j < nFaceAttributes; ++j)
            {
                faceFS >> _;
            }
        }

        // 3. .sample.ply file
        std::fstream plyFS(filename + ".sample.ply");
        assert(plyFS.is_open() && "tetgen_loader::ERROR .sample.ply file not found.");
        unsigned int nSamples;
        std::string header;
        for (;;)
        {
            plyFS >> header;
            if (header == "element")
            {
                plyFS >> header;
                if (header == "vertex")
                {
                    plyFS >> nSamples;
                }
            }
            else if (header == "end_header")
            {
                break;
            }
        }
        sample_tri_idx.resize(nSamples);
        sample_barycentric_weights.resize(nSamples * 3);
        float x, y, z, tri_id, b0, b1, b2;
        for (unsigned int i = 0; i < nSamples; ++i)
        {
            unsigned int _;
            plyFS >> x >> y >> z >> tri_id >> _ >> b0 >> b1 >> b2;
            sample_tri_idx[i] = (unsigned int)tri_id;
            sample_barycentric_weights[3 * i] = b0;
            sample_barycentric_weights[3 * i + 1] = b1;
            sample_barycentric_weights[3 * i + 2] = b2;
        }
    }
}

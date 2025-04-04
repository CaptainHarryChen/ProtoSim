#include "TetrahedronLoader.h"
#include <fstream>
#include <cassert>
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

        // 2. .face file
        std::fstream faceFS(filename + ".face");
        assert(faceFS.is_open() && "tetgen_loader::ERROR .face file not found.");
        // <# of faces> <nodes per face> <# of attributes>
        unsigned int nFaces, nFaceAttribs;
        faceFS >> nFaces >> nFaceAttribs;
        surface_triangles.resize(nFaces * 3);
        for (unsigned int i = 0; i < nFaces; ++i)
        {
            int _;
            faceFS >> _ >> surface_triangles[3 * i] >> surface_triangles[3 * i + 1] >> surface_triangles[3 * i + 2];
            if (nodeStartAtZero == false)
            {
                surface_triangles[3 * i] -= 1;
                surface_triangles[3 * i + 1] -= 1;
                surface_triangles[3 * i + 2] -= 1;
            }
            for (unsigned int j = 0; j < nFaceAttribs; ++j)
            {
                faceFS >> _;
                if (_ == -1)
                {
                    unsigned int t = surface_triangles[3 * i];
                    surface_triangles[3 * i] = surface_triangles[3 * i + 1];
                    surface_triangles[3 * i + 1] = t;
                }
            }
        }
    }
}

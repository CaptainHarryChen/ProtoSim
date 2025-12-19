#pragma once
#include <string>
#include <vector>
#include <glm/glm.hpp>

namespace viewer
{
    namespace TetrahedronLoader
    {
        void LoadTetrahedron(
            std::string filename,
            float scale, glm::vec3 translate, glm::vec3 rotate,
            std::vector<float> &vertices, std::vector<unsigned int> &surface_triangles, std::vector<unsigned int> &tetrahedras);

        void LoadTetrahedronWithPLYSample(
            std::string filename,
            float scale, glm::vec3 translate, glm::vec3 rotate,
            std::vector<float> &vertices, std::vector<unsigned int> &surface_triangles, std::vector<unsigned int> &tetrahedras,
            std::vector<unsigned int> &sample_tri_idx, std::vector<float> &sample_barycentric_weights);
    }
} // namespace viewer

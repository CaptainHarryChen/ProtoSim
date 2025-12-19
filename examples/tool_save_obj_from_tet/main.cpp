#include <fstream>
#include <iostream>
#include <vector>
#include <common.h>

int main()
{
    std::vector<float> positions;
    std::vector<unsigned int> surface_triangles;
    std::vector<unsigned int> tetrahedras;
    viewer::TetrahedronLoader::LoadTetrahedron(std::string(ASSET_DIR) + "/bunny", 1.0f, glm::vec3(0.0f), glm::vec3(0.0f), positions, surface_triangles, tetrahedras);
    std::ofstream obj_file(std::string(ASSET_DIR) + "/bunny.obj");
    for (size_t i = 0; i < positions.size() / 3; ++i)
    {
        obj_file << "v " << positions[i * 3 + 0] << " " << positions[i * 3 + 1] << " " << positions[i * 3 + 2] << "\n";
    }
    for (size_t i = 0; i < surface_triangles.size() / 3; ++i)
    {
        obj_file << "f " << surface_triangles[i * 3 + 0] + 1 << " " << surface_triangles[i * 3 + 1] + 1 << " " << surface_triangles[i * 3 + 2] + 1 << "\n";
    }
    obj_file.close();

    return 0;
}

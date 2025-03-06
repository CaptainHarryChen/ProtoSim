#pragma once
#include <vector>
#include <Mesh/MeshBase.h>

class Floor : public MeshBase
{
public:
    Floor(float scale = 1.0, bool isCubic = true);

private:
    static std::vector<Vertex> __PLANE_VERTICES;
    static std::vector<unsigned int> __PLANE_INDICES;
};

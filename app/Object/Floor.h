#pragma once
#include <vector>
#include <memory>
#include <Object/Object.h>
#include <Mesh/Mesh.h>

class Floor : public Object
{
public:
    Floor(float scale = 1.0);

    std::shared_ptr<Mesh> GetMesh();

protected:
    std::shared_ptr<Mesh> m_mesh;

private:
    static std::vector<Vertex> __PLANE_VERTICES;
    static std::vector<unsigned int> __PLANE_INDICES;
};

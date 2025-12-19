#pragma once
#include <vector>
#include <memory>
#include <viewer/Framework/Object.h>
#include <viewer/RenderObject/Mesh.h>

namespace viewer {

class Floor : public Object
{
public:
    Floor(float scale = 1.0);

    std::shared_ptr<Mesh> GetMesh();

protected:
    std::shared_ptr<Mesh> m_mesh;

private:
    static std::vector<Vertex>       __PLANE_VERTICES;
    static std::vector<unsigned int> __PLANE_INDICES;
};

}  // namespace viewer

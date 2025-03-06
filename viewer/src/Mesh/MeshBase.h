#pragma once
#include <Mesh/SurfaceMesh.h>

class MeshBase
{
public:
	MeshBase() = default;
	virtual ~MeshBase() = default;

	virtual std::shared_ptr<SurfaceMesh> getMesh();

	std::shared_ptr<SurfaceMesh> mesh;
};

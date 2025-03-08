#pragma once
#include <Mesh/Mesh.h>

/// @brief Base class for all mesh objects.
class MeshBase
{
public:
	MeshBase() = default;
	virtual ~MeshBase() = default;

	virtual std::shared_ptr<Mesh> getMesh();

protected:
	std::shared_ptr<Mesh> mesh;
};

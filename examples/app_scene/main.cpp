#include "scene/scene.hpp"
#include "scene/simulation_scene.hpp"
#include "proj_config.h"
#include "viewer_config.h"
#include "mesh_IO/tetgen_loader.hpp"
#include "mesh_IO/gmsh_to_tetgen.hpp"
#include "solver/example_solver.hpp"

#include <string>
int main()
{
	// std::vector<float> vertices;
    // std::vector<unsigned int> surface_triangles;
    // std::vector<unsigned int> tetrahedras;
	// gmsh_tet_loader(std::string(ASSET_DIR) + "/bunny.msh", vertices, surface_triangles, tetrahedras);
	// std::cout << vertices.size() / 3 << " " << surface_triangles.size() / 3 << " " << tetrahedras.size() / 4 << std::endl;
	// tetgen_loader(std::string(ASSET_DIR) + "/bunny.1", vertices, surface_triangles, tetrahedras);
	// gmsh_to_tetgen(std::string(ASSET_DIR) + std::string("/bunny.msh"), std::string(ASSET_DIR) + std::string("/bunny"));
	std::shared_ptr<example_solver<double>> solver = std::make_shared<example_solver<double>>();
	simulation_scene<double> sim("test", false);
	sim.bind_solver(solver);
	//sim.loadSurfaceMeshFromOBJ(
	//	std::string(VIEWER_DIR) + "/data/objs/venus.obj", 0.01,
	//	glm::vec3(-1.0, 0.0, 0.0), glm::vec3(glm::radians(-90.0f), 0., 0.),
	//	{ glm::vec3(1.), glm::vec3(0.1, 0.1, 0.1) });
	//sim.loadSurfaceMeshFromOBJ(
	//	std::string(VIEWER_DIR) + "/data/objs/venus.obj", 0.01,
	//	glm::vec3(1.0, 0.0, 0.0), glm::vec3(glm::radians(-90.0f), 0., 0.),
	//	{ glm::vec3(1.), glm::vec3(0.1, 0.1, 0.1) });
	//sim.loadSurfaceMeshFromOBJ(std::string(VIEWER_DIR)
	//	+ std::string("/data/objs/CartoonSquirelModel/CartoonSquirelModel.obj"),
	//	0.5, glm::vec3(0., 0., -1.), glm::vec3(0.),
	//	std::string(VIEWER_DIR) + std::string("/data/objs/CartoonSquirelModel/DiffSqurel.tga"));
	sim.loadTriangleMesh(std::string(VIEWER_DIR) + "/data/objs/bunny.obj", 0u, 3.0, glm::vec3(0., 2., 0.), glm::vec3(glm::radians(90.f), 0., 0.),
		{ glm::vec3(1.), glm::vec3(0.1) });
	sim.loadTetrahedraMesh(std::string(ASSET_DIR) + std::string("/bunny/bunny.1"), 0, // 0: tetgen, 1: gmsh
		1u, 3.0, glm::vec3(0., 2., 0.), glm::vec3(glm::radians(90.f), 0., 0.));
	//sim.loadStaticParticles(glm::vec3(2.), glm::vec3(3.), 0.1, 0.05f, glm::vec3(0., 0., 1.));
	//sim.loadParticles(glm::vec3(4., 2., 2.), glm::vec3(5., 3., 3.), 0.1, 1u, 0.05f, glm::vec3(0., 1.0, 0.));
	//sim.loadParticles(glm::vec3(6., 2., 2.), glm::vec3(7., 3., 3.), 0.1, 0u);
	//sim.particleColorMapping = true;
	sim.run();
	return 0;
}

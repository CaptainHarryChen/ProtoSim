#pragma once
#include "scene.hpp"
#include <iostream>
#include "solver/solver_base.hpp"
#include "geometric_tools/glm_transform_matrix.hpp"
#include "geometric_tools/verts_to_render_buffer.hpp"
#include "mesh_IO/obj_loader.hpp"
#include "mesh_IO/tetgen_loader.hpp"
#include "mesh_IO/gmsh_tet_loader.hpp"
#include "mesh_IO/ipc_msh_tet_loader.hpp"

template<typename Real>
class simulation_scene : public scene
{
public:
    simulation_scene(const char* _name = "Simulation", const bool _isShadowMappingCubic = false);
    
    void bind_solver(std::shared_ptr<solver_base<Real>> solver);

    void loadTriangleMesh(std::string filename, unsigned int isDynamic = 0u,
        float scale = 1.0, glm::vec3 translate = glm::vec3(0.), glm::vec3 rotate = glm::vec3(0.),
        std::vector<glm::vec3> material = { glm::vec3(.5, .5, 1.), glm::vec3(.1) });

    void loadTriangleMesh(std::string filename, unsigned int isDynamic,
        glm::vec3 scale = glm::vec3(1.), glm::vec3 translate = glm::vec3(0.), glm::vec3 rotate = glm::vec3(0.),
        std::vector<glm::vec3> material = { glm::vec3(.5, .5, 1.), glm::vec3(.1) });
    
    void loadTetrahedraMesh(std::string filename, unsigned int tetfile_type = 0, // 0: tetgen, 1: gmsh
        unsigned int isDynamic = 0u,
        float scale = 1.0, glm::vec3 translate = glm::vec3(0.), glm::vec3 rotate = glm::vec3(0.),
        std::vector<glm::vec3> material = { glm::vec3(.5, .5, 1.), glm::vec3(.1) });

    void loadTetrahedraMesh(std::string filename, unsigned int tetfile_type = 0, // 0: tetgen, 1: gmsh
        unsigned int isDynamic = 0u,
        glm::vec3 scale = glm::vec3(1.), glm::vec3 translate = glm::vec3(0.), glm::vec3 rotate = glm::vec3(0.),
        std::vector<glm::vec3> material = { glm::vec3(.5, .5, 1.), glm::vec3(.1) });

    void loadParticles(glm::vec3 lower_bound, glm::vec3 upper_bound, float delta, unsigned int isDynamic = 0u,
        float radius = 0.05, glm::vec3 color = glm::vec3(1.f), glm::vec2 material = glm::vec2(.1));

    void loadParticles_pertubation(glm::vec3 lower_bound, glm::vec3 upper_bound, float delta, unsigned int isDynamic = 0u,
        float radius = 0.05, glm::vec3 color = glm::vec3(1.f), glm::vec2 material = glm::vec2(.1));

    void loadFloor(glm::vec3 lower_left, float x_len, float z_len,
        std::vector<glm::vec3> material = { glm::vec3(.5, .5, 1.), glm::vec3(.1) });

    void updateTriangleMeshes();
    void updateParticles(bool withColor = false);

    void lightingSetUp();

    void run();

    bool particleColorMapping = false;
protected:
    std::vector<std::shared_ptr<SurfaceMesh>> m_render_meshes;
    std::vector<Real> m_positions; std::vector<unsigned int> m_positions_offsets;
    std::vector<unsigned int> m_surface_triangles; std::vector<unsigned int> m_surface_triangles_offsets;
    std::vector<unsigned int> m_tetrahedras; std::vector<unsigned int> m_tetrahedras_offsets;
    std::vector<unsigned int> m_isObjectDynamic;

    std::vector<std::shared_ptr<Spheres>> m_render_particles; std::vector<glm::vec3> m_colors_particles;
    std::vector<Real> m_particles_positions; std::vector<unsigned int> m_particles_positions_offsets;
    std::vector<unsigned int> m_areParticlesDynamic;

    std::shared_ptr<solver_base<Real>> m_solver;
    glm::vec3 rotateVecQuat(glm::quat q, glm::vec3 v);
public:
    void makeSpaceForParticles(unsigned int nParticles, unsigned int isDynamic = 0u,
        float radius = 0.05, glm::vec3 color = glm::vec3(1.f), glm::vec2 material = glm::vec2(.1));
};

#include "simulation_scene.impl.hpp"
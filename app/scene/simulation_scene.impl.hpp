#include "simulation_scene.hpp"
#include <random>
#include <time.h>
#include <chrono>

template<typename Real>
simulation_scene<Real>::simulation_scene(const char* _name, const bool _isShadowMappingCubic) : scene(_name, _isShadowMappingCubic)
{
    name = std::string(_name);
    m_positions_offsets.push_back(0);
    m_surface_triangles_offsets.push_back(0);
    m_tetrahedras_offsets.push_back(0);
    m_particles_positions_offsets.push_back(0);
    m_solver = std::make_shared<solver_base<Real>>();
}

template<typename Real>
void simulation_scene<Real>::bind_solver(std::shared_ptr<solver_base<Real>> solver)
{
    m_solver = solver;
}

template<typename Real>
void simulation_scene<Real>::loadFloor(glm::vec3 lower_left, float x_len, float z_len,
    std::vector<glm::vec3> material)
{
    unsigned int nVerts = 4;
    unsigned int nTris = 2;
    std::vector<Vertex> render_buffer;
    float x = lower_left.x;
    float y = lower_left.y;
    float z = lower_left.z;
    std::vector<float> vertices =
    {
        x, y, z + z_len,
        x + x_len, y, z + z_len,
        x, y, z,
        x + x_len, y, z
    };
    std::vector<unsigned int> indices =
    {
        0, 1, 2,
        1, 3, 2
    };

    unsigned int positions_start_at = m_positions_offsets[m_positions_offsets.size() - 1];
    unsigned int surface_triangles_start_at = m_surface_triangles_offsets[m_surface_triangles_offsets.size() - 1];
    unsigned int tetrahedras_start_at = m_tetrahedras_offsets[m_tetrahedras_offsets.size() - 1];
    m_positions_offsets.push_back(positions_start_at + vertices.size());
    m_surface_triangles_offsets.push_back(surface_triangles_start_at + indices.size());
    m_tetrahedras_offsets.push_back(tetrahedras_start_at);

    for (unsigned int i = 0; i < nVerts; ++i)
    {
        m_positions.push_back(vertices[3 * i]);
        m_positions.push_back(vertices[3 * i + 1]);
        m_positions.push_back(vertices[3 * i + 2]);
    }

    for (unsigned int i = 0; i < nTris; ++i)
    {
        m_surface_triangles.push_back(positions_start_at / 3 + indices[3 * i]);
        m_surface_triangles.push_back(positions_start_at / 3 + indices[3 * i + 1]);
        m_surface_triangles.push_back(positions_start_at / 3 + indices[3 * i + 2]);
    }

    verts_to_render_buffer(vertices, render_buffer);
    std::shared_ptr<SurfaceMesh> surface_mesh = std::make_shared<SurfaceMesh>(render_buffer, indices, material, isShadowMappingCubic);
    m_isObjectDynamic.push_back(false);
    m_render_meshes.push_back(surface_mesh);
}

template<typename Real>
void simulation_scene<Real>::loadTriangleMesh(std::string filename, unsigned int isDynamic,
    float scale, glm::vec3 translate, glm::vec3 rotate,
    std::vector<glm::vec3> material)
{
    unsigned int nVerts;
    unsigned int nTris;
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    std::vector<Vertex> render_buffer;
    obj_loader(filename, vertices, indices);
    // glm::mat4 transform = glm::transform_matrix(scale, translate, rotate);
    nVerts = vertices.size() / 3;
    nTris = indices.size() / 3;

    unsigned int positions_start_at = m_positions_offsets[m_positions_offsets.size() - 1];
    unsigned int surface_triangles_start_at = m_surface_triangles_offsets[m_surface_triangles_offsets.size() - 1];
    unsigned int tetrahedras_start_at = m_tetrahedras_offsets[m_tetrahedras_offsets.size() - 1];
    m_positions_offsets.push_back(positions_start_at + vertices.size());
    m_surface_triangles_offsets.push_back(surface_triangles_start_at + indices.size());
    m_tetrahedras_offsets.push_back(tetrahedras_start_at);

    glm::quat q(rotate);
    for (unsigned int i = 0; i < nVerts; ++i)
    {
        // glm::vec3 transformed_verts = transform * glm::vec4(vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2], 1.);
        // vertices[3 * i] = transformed_verts.x;
        // vertices[3 * i + 1] = transformed_verts.y;
        // vertices[3 * i + 2] = transformed_verts.z;
        // m_positions.push_back(transformed_verts.x);
        // m_positions.push_back(transformed_verts.y);
        // m_positions.push_back(transformed_verts.z);
        glm::vec3 __vertex(
            vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2]
        );
        __vertex = rotateVecQuat(q, __vertex);
        __vertex = scale * __vertex + translate;
        vertices[3 * i] = __vertex.x; vertices[3 * i + 1] = __vertex.y; vertices[3 * i + 2] = __vertex.z;
        m_positions.push_back(__vertex.x);
        m_positions.push_back(__vertex.y);
        m_positions.push_back(__vertex.z);
    }

    for (unsigned int i = 0; i < nTris; ++i)
    {
        m_surface_triangles.push_back(positions_start_at / 3 + indices[3 * i]);
        m_surface_triangles.push_back(positions_start_at / 3 + indices[3 * i + 1]);
        m_surface_triangles.push_back(positions_start_at / 3 + indices[3 * i + 2]);
    }

    verts_to_render_buffer(vertices, render_buffer);
    std::shared_ptr<SurfaceMesh> surface_mesh = std::make_shared<SurfaceMesh>(render_buffer, indices, material, isShadowMappingCubic);
    m_isObjectDynamic.push_back(isDynamic);
    m_render_meshes.push_back(surface_mesh);
}

template<typename Real>
void simulation_scene<Real>::loadTriangleMesh(std::string filename, unsigned int isDynamic,
    glm::vec3 scale, glm::vec3 translate, glm::vec3 rotate,
    std::vector<glm::vec3> material)
{
    unsigned int nVerts;
    unsigned int nTris;
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    std::vector<Vertex> render_buffer;
    obj_loader(filename, vertices, indices);
    // glm::mat4 transform = glm::transform_matrix(scale, translate, rotate);
    nVerts = vertices.size() / 3;
    nTris = indices.size() / 3;

    unsigned int positions_start_at = m_positions_offsets[m_positions_offsets.size() - 1];
    unsigned int surface_triangles_start_at = m_surface_triangles_offsets[m_surface_triangles_offsets.size() - 1];
    unsigned int tetrahedras_start_at = m_tetrahedras_offsets[m_tetrahedras_offsets.size() - 1];
    m_positions_offsets.push_back(positions_start_at + vertices.size());
    m_surface_triangles_offsets.push_back(surface_triangles_start_at + indices.size());
    m_tetrahedras_offsets.push_back(tetrahedras_start_at);

    glm::quat q(rotate);
    for (unsigned int i = 0; i < nVerts; ++i)
    {
        // glm::vec3 transformed_verts = transform * glm::vec4(vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2], 1.);
        // vertices[3 * i] = transformed_verts.x;
        // vertices[3 * i + 1] = transformed_verts.y;
        // vertices[3 * i + 2] = transformed_verts.z;
        // m_positions.push_back(transformed_verts.x);
        // m_positions.push_back(transformed_verts.y);
        // m_positions.push_back(transformed_verts.z);
        glm::vec3 __vertex(
            vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2]
        );
        __vertex = rotateVecQuat(q, __vertex);
        __vertex = glm::vec3(scale.x * __vertex.x, scale.y * __vertex.y, scale.z * __vertex.z) + translate;
        vertices[3 * i] = __vertex.x; vertices[3 * i + 1] = __vertex.y; vertices[3 * i + 2] = __vertex.z;
        m_positions.push_back(__vertex.x);
        m_positions.push_back(__vertex.y);
        m_positions.push_back(__vertex.z);
    }

    for (unsigned int i = 0; i < nTris; ++i)
    {
        m_surface_triangles.push_back(positions_start_at / 3 + indices[3 * i]);
        m_surface_triangles.push_back(positions_start_at / 3 + indices[3 * i + 1]);
        m_surface_triangles.push_back(positions_start_at / 3 + indices[3 * i + 2]);
    }

    verts_to_render_buffer(vertices, render_buffer);
    std::shared_ptr<SurfaceMesh> surface_mesh = std::make_shared<SurfaceMesh>(render_buffer, indices, material, isShadowMappingCubic);
    m_isObjectDynamic.push_back(isDynamic);
    m_render_meshes.push_back(surface_mesh);
}

template<typename Real>
void simulation_scene<Real>::loadTetrahedraMesh(std::string filename, unsigned int tetfile_type,
    unsigned int isDynamic,
    float scale, glm::vec3 translate, glm::vec3 rotate,
    std::vector<glm::vec3> material)
{
    unsigned int nVerts;
    unsigned int nSurfaceTris;
    unsigned int nTets;
    std::vector<float> vertices;
    std::vector<unsigned int> surface_triangles;
    std::vector<unsigned int> tetrahedras;
    std::vector<Vertex> render_buffer;
    if (tetfile_type == 0) tetgen_loader(filename, vertices, surface_triangles, tetrahedras);
    else if (tetfile_type == 1) gmsh_tet_loader(filename, vertices, surface_triangles, tetrahedras);
    else if (tetfile_type == 2) ipc_msh_tet_loader(filename, vertices, surface_triangles, tetrahedras);
    else
    {
        printf("loadTetrahedraMesh::N/A\n");
        exit(1);
    }
    // glm::mat4 transform = glm::transform_matrix(scale, translate, rotate);
    nVerts = vertices.size() / 3;
    nSurfaceTris = surface_triangles.size() / 3;
    nTets = tetrahedras.size() / 4;

    unsigned int positions_start_at = m_positions_offsets[m_positions_offsets.size() - 1];
    unsigned int surface_triangles_start_at = m_surface_triangles_offsets[m_surface_triangles_offsets.size() - 1];
    unsigned int tetrahedras_start_at = m_tetrahedras_offsets[m_tetrahedras_offsets.size() - 1];
    m_positions_offsets.push_back(positions_start_at + vertices.size());
    m_surface_triangles_offsets.push_back(surface_triangles_start_at + surface_triangles.size());
    m_tetrahedras_offsets.push_back(tetrahedras_start_at + tetrahedras.size());

    glm::quat q(rotate);
    for(unsigned int i = 0; i < nVerts; ++i)
    {
        // glm::vec3 transformed_verts = transform * glm::vec4(vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2], 1.);
        // vertices[3 * i] = transformed_verts.x;
        // vertices[3 * i + 1] = transformed_verts.y;
        // vertices[3 * i + 2] = transformed_verts.z;
        // m_positions.push_back(transformed_verts.x);
        // m_positions.push_back(transformed_verts.y);
        // m_positions.push_back(transformed_verts.z);
        glm::vec3 __vertex(
            vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2]
        );
        __vertex = rotateVecQuat(q, __vertex);
        __vertex = scale * __vertex + translate;
        vertices[3 * i] = __vertex.x; vertices[3 * i + 1] = __vertex.y; vertices[3 * i + 2] = __vertex.z;
        m_positions.push_back(__vertex.x);
        m_positions.push_back(__vertex.y);
        m_positions.push_back(__vertex.z);
    }

    for (unsigned int i = 0; i < nSurfaceTris; ++i)
    {
        m_surface_triangles.push_back(positions_start_at / 3 + surface_triangles[3 * i]);
        m_surface_triangles.push_back(positions_start_at / 3 + surface_triangles[3 * i + 1]);
        m_surface_triangles.push_back(positions_start_at / 3 + surface_triangles[3 * i + 2]);
    }

    for (unsigned int i = 0; i < nTets; ++i)
    {
        m_tetrahedras.push_back(positions_start_at / 3 + tetrahedras[4 * i]);
        m_tetrahedras.push_back(positions_start_at / 3 + tetrahedras[4 * i + 1]);
        m_tetrahedras.push_back(positions_start_at / 3 + tetrahedras[4 * i + 2]);
        m_tetrahedras.push_back(positions_start_at / 3 + tetrahedras[4 * i + 3]);
    }

    verts_to_render_buffer(vertices, render_buffer);
    std::shared_ptr<SurfaceMesh> surface_mesh = std::make_shared<SurfaceMesh>(render_buffer, surface_triangles, material, isShadowMappingCubic);
    m_isObjectDynamic.push_back(isDynamic);
    m_render_meshes.push_back(surface_mesh);
}

template<typename Real>
void simulation_scene<Real>::loadTetrahedraMesh(std::string filename, unsigned int tetfile_type,
    unsigned int isDynamic,
    glm::vec3 scale, glm::vec3 translate, glm::vec3 rotate,
    std::vector<glm::vec3> material)
{
    unsigned int nVerts;
    unsigned int nSurfaceTris;
    unsigned int nTets;
    std::vector<float> vertices;
    std::vector<unsigned int> surface_triangles;
    std::vector<unsigned int> tetrahedras;
    std::vector<Vertex> render_buffer;
    if (tetfile_type == 0) tetgen_loader(filename, vertices, surface_triangles, tetrahedras);
    else if (tetfile_type == 1) gmsh_tet_loader(filename, vertices, surface_triangles, tetrahedras);
    else if (tetfile_type == 2) ipc_msh_tet_loader(filename, vertices, surface_triangles, tetrahedras);
    else
    {
        printf("loadTetrahedraMesh::N/A\n");
        exit(1);
    }
    nVerts = vertices.size() / 3;
    nSurfaceTris = surface_triangles.size() / 3;
    nTets = tetrahedras.size() / 4;

    unsigned int positions_start_at = m_positions_offsets[m_positions_offsets.size() - 1];
    unsigned int surface_triangles_start_at = m_surface_triangles_offsets[m_surface_triangles_offsets.size() - 1];
    unsigned int tetrahedras_start_at = m_tetrahedras_offsets[m_tetrahedras_offsets.size() - 1];
    m_positions_offsets.push_back(positions_start_at + vertices.size());
    m_surface_triangles_offsets.push_back(surface_triangles_start_at + surface_triangles.size());
    m_tetrahedras_offsets.push_back(tetrahedras_start_at + tetrahedras.size());

    glm::quat q(rotate);
    for (unsigned int i = 0; i < nVerts; ++i)
    {
        glm::vec3 __vertex(
            vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2]
        );
        __vertex = rotateVecQuat(q, __vertex);
        __vertex = glm::vec3(scale.x * vertices[3 * i], scale.y * vertices[3 * i + 1], scale.z * vertices[3 * i + 2]) + translate;
        vertices[3 * i] = __vertex.x; vertices[3 * i + 1] = __vertex.y; vertices[3 * i + 2] = __vertex.z;
        m_positions.push_back(__vertex.x);
        m_positions.push_back(__vertex.y);
        m_positions.push_back(__vertex.z);
    }

    for (unsigned int i = 0; i < nSurfaceTris; ++i)
    {
        m_surface_triangles.push_back(positions_start_at / 3 + surface_triangles[3 * i]);
        m_surface_triangles.push_back(positions_start_at / 3 + surface_triangles[3 * i + 1]);
        m_surface_triangles.push_back(positions_start_at / 3 + surface_triangles[3 * i + 2]);
    }

    for (unsigned int i = 0; i < nTets; ++i)
    {
        m_tetrahedras.push_back(positions_start_at / 3 + tetrahedras[4 * i]);
        m_tetrahedras.push_back(positions_start_at / 3 + tetrahedras[4 * i + 1]);
        m_tetrahedras.push_back(positions_start_at / 3 + tetrahedras[4 * i + 2]);
        m_tetrahedras.push_back(positions_start_at / 3 + tetrahedras[4 * i + 3]);
    }

    verts_to_render_buffer(vertices, render_buffer);
    std::shared_ptr<SurfaceMesh> surface_mesh = std::make_shared<SurfaceMesh>(render_buffer, surface_triangles, material, isShadowMappingCubic);
    m_isObjectDynamic.push_back(isDynamic);
    m_render_meshes.push_back(surface_mesh);
}

template<typename Real>
void simulation_scene<Real>::loadParticles(glm::vec3 lower_bound, glm::vec3 upper_bound, float delta,
    unsigned int isDynamic, float radius, glm::vec3 color, glm::vec2 material)
{
    std::vector<Point> points;
    m_colors_particles.push_back(color);
    for (float x = lower_bound.x; x <= upper_bound.x; x += delta) for (float y = lower_bound.y; y <= upper_bound.y; y += delta) for (float z = lower_bound.z; z <= upper_bound.z; z += delta)
    {
        points.push_back({ glm::vec3(x, y, z), color });
        m_particles_positions.push_back(x);
        m_particles_positions.push_back(y);
        m_particles_positions.push_back(z);
    }

    unsigned int positions_start_at = m_particles_positions_offsets[m_particles_positions_offsets.size() - 1];
    m_particles_positions_offsets.push_back(positions_start_at + m_particles_positions.size());
    m_areParticlesDynamic.push_back(isDynamic);

    std::shared_ptr<Spheres> spheres = std::make_shared<Spheres>(points, material);
    spheres->radius = radius;
    particles.push_back(spheres);
    m_render_particles.push_back(spheres);
}

template<typename Real>
void simulation_scene<Real>::loadParticles_pertubation(glm::vec3 lower_bound, glm::vec3 upper_bound, float delta,
    unsigned int isDynamic, float radius, glm::vec3 color, glm::vec2 material)
{
    srand(time(NULL));
    std::vector<Point> points;
    m_colors_particles.push_back(color);
    Real pr = 0.1;
    for (float x = lower_bound.x; x <= upper_bound.x; x += delta * (1. + 2. * pr)) for (float y = lower_bound.y; y <= upper_bound.y; y += delta * (1. + 2. * pr)) for (float z = lower_bound.z; z <= upper_bound.z; z += delta * (1. + 2. * pr))
    {
        Real r1 = (Real(rand()) - Real(.5 * RAND_MAX)) / Real(RAND_MAX) * 2 * delta * pr;
        Real r2 = (Real(rand()) - Real(.5 * RAND_MAX)) / Real(RAND_MAX) * 2 * delta * pr;
        Real r3 = (Real(rand()) - Real(.5 * RAND_MAX)) / Real(RAND_MAX) * 2 * delta * pr;
        //std::cout << r1 << " " << r2 << " " << r3 << std::endl;
        points.push_back({ glm::vec3(x + r1, y + r2, z + r3), color });
        m_particles_positions.push_back(x + r1);
        m_particles_positions.push_back(y + r2);
        m_particles_positions.push_back(z + r3);
    }

    unsigned int positions_start_at = m_particles_positions_offsets[m_particles_positions_offsets.size() - 1];
    m_particles_positions_offsets.push_back(positions_start_at + m_particles_positions.size());
    m_areParticlesDynamic.push_back(isDynamic);

    std::shared_ptr<Spheres> spheres = std::make_shared<Spheres>(points, material);
    spheres->radius = radius;
    particles.push_back(spheres);
    m_render_particles.push_back(spheres);
}

template<typename Real>
void simulation_scene<Real>::updateTriangleMeshes()
{
    //glEnable(GL_CULL_FACE);
    //glCullFace(GL_BACK);
    for (unsigned int i = 0; i < m_render_meshes.size(); ++i)
    {
        //if (m_isObjectDynamic[i] == 1)
        {
            std::vector<Vertex> update_data;
            verts_to_render_buffer(&(m_solver->get_positions())[m_positions_offsets[i]], update_data, m_render_meshes[i]->vertices.size());
            m_render_meshes[i]->UpdateVertices(update_data);
        }
    }
    //glDisable(GL_CULL_FACE);
}

template<typename Real>
void simulation_scene<Real>::updateParticles(bool withColor)
{
    for (unsigned int i = 0; i < m_render_particles.size(); ++i)
    {
        if (m_areParticlesDynamic[i] == 1)
        {
            std::vector<Point> update_data;
            if (withColor == false)
            {
                verts_to_render_buffer_particles(
                    &(m_solver->get_particles_positions())[m_particles_positions_offsets[i]],
                    update_data, m_render_particles[i]->vertices.size(), m_colors_particles[i]
                );
            }
            else
            {
                verts_to_render_buffer_particles(
                    &(m_solver->get_particles_positions())[m_particles_positions_offsets[i]],
                    update_data, m_render_particles[i]->vertices.size(), m_solver->get_particles_colors()
                );
            }
            m_render_particles[i]->UpdateVertices(update_data);
        }
    }
}

template<typename Real>
void simulation_scene<Real>::lightingSetUp()
{
    if (fakeFloor)
    {
        floor = std::make_shared<Floor>(100.0f, isShadowMappingCubic);
        meshes.push_back(floor->getMesh());
    }
    for(unsigned int i = 0; i < static_mesh_loaders.size(); ++i)
        meshes.push_back(static_mesh_loaders[i].getMesh());
    for(unsigned int i = 0; i < m_render_meshes.size(); ++i)
        meshes.push_back(m_render_meshes[i]);
    lights.push_back(Light(glm::vec3(-3.0f, 3.0f, -3.0f), glm::vec3(1.0, 0.0, 0.0), meshes, isShadowMappingCubic));
    lights.push_back(Light(glm::vec3(-3.0f, 3.0f, 3.0f), glm::vec3(0.0, 1.0, 0.0), meshes, isShadowMappingCubic));
    lights.push_back(Light(glm::vec3(3.0f, 3.0f, -3.0f), glm::vec3(0.0, 0.0, 1.0), meshes, isShadowMappingCubic));
    lights.push_back(Light(glm::vec3(3.0f, 3.0f, 3.0f), glm::vec3(1.0, 1.0, 1.0), meshes, isShadowMappingCubic));
}
    
template<typename Real>
void simulation_scene<Real>::run()
{
    std::cout << "simulation_scene<Real>::run\n";
    lightingSetUp();
    m_solver->initialization(m_positions, m_positions_offsets, m_surface_triangles, m_surface_triangles_offsets,
        m_tetrahedras, m_tetrahedras_offsets, m_isObjectDynamic,
        m_particles_positions, m_particles_positions_offsets, m_areParticlesDynamic);
    while (!glfwWindowShouldClose(window))
    {
        m_solver->step();

        glfwPollEvents();
        camera->processInput(window);

        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClearColor(clearColor.x, clearColor.y, clearColor.z, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        camera->computeMVP(model, view, projection);
        drawLights();
        updateTriangleMeshes();
        drawMeshes();
        updateParticles(particleColorMapping);
        drawLines();
        drawParticles();
        updateLighting();
        
        scene::setImGUI();

        glfwSwapBuffers(window);
    }
}

template<typename Real>
glm::vec3 simulation_scene<Real>::rotateVecQuat(glm::quat q, glm::vec3 v)
{
    glm::vec3 u(q.x, q.y, q.z);
    float s = q.w;
    return 2.0f * glm::dot(u, v) * u + (s * s - glm::dot(u, u)) * v + 2.0f * s * glm::cross(u, v);
}

template<typename Real>
void simulation_scene<Real>::makeSpaceForParticles(unsigned int nParticles, unsigned int isDynamic,
    float radius, glm::vec3 color, glm::vec2 material)
{
    std::vector<Point> points;
    m_colors_particles.push_back(color);
    for (unsigned int i = 0; i < nParticles; ++i)
    {
        points.push_back({ glm::vec3(FLT_MAX), color });
        m_particles_positions.push_back(FLT_MAX);
        m_particles_positions.push_back(FLT_MAX);
        m_particles_positions.push_back(FLT_MAX);
    }

    unsigned int positions_start_at = m_particles_positions_offsets[m_particles_positions_offsets.size() - 1];
    m_particles_positions_offsets.push_back(positions_start_at + m_particles_positions.size());
    m_areParticlesDynamic.push_back(isDynamic);

    std::shared_ptr<Spheres> spheres = std::make_shared<Spheres>(points, material);
    spheres->radius = radius;
    particles.push_back(spheres);
    m_render_particles.push_back(spheres);
}
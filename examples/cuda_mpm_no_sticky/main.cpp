#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <GLFWApp.h>
#include <Object/CubeLineBox.h>
#include <Scene/SimulationScene.h>
#include <Geometry/ParticleBatch.h>
#include <Geometry/LineSegment.h>
#include <Geometry/SphereRenderer.h>
#include <Render/RenderSystem.h>
#include <Mesh/SolidColorRenderer.h>
#include <glm/gtc/quaternion.hpp>
#include <tiny_obj_loader.h>
#include <proj_config.h>
#include "IQMPMSolver.cuh"
#include "IQMPMDebugConnector.cuh"

static glm::vec3 rotateVecQuat(glm::quat q, glm::vec3 v)
{
    glm::vec3 u(q.x, q.y, q.z);
    float s = q.w;
    return 2.0f * glm::dot(u, v) * u + (s * s - glm::dot(u, u)) * v + 2.0f * s * glm::cross(u, v);
}

template <typename Real>
class IQMPMScene : public SimulationScene<Real>
{
public:
    std::vector<unsigned int> m_object_types;
    std::vector<unsigned int> m_particle_object_id;
    std::vector<Real> m_particle_masses;
    std::vector<Real> m_particle_volumes;

    virtual void AddMPMCubeParticleBatch(glm::vec3 lower_bound, glm::vec3 upper_bound, float dis,
                                         unsigned int particle_type, Real total_mass,
                                         float radius, glm::vec3 color, glm::vec2 material)
    {
        m_object_types.push_back(particle_type);
        unsigned int object_id = (unsigned int)m_object_types.size() - 1;
        SimulationScene<Real>::AddCubeParticleBatch(lower_bound, upper_bound, dis, radius, color, material);
        size_t num_particle = this->m_particle_batch_offsets.back().first->m_particles.size();
        Real total_volume = (upper_bound.x - lower_bound.x) * (upper_bound.y - lower_bound.y) * (upper_bound.z - lower_bound.z);
        Real particle_volume = dis * dis * dis;
        Real particle_mass = total_mass * particle_volume / total_volume;
        for (size_t i = 0; i < num_particle; ++i)
        {
            m_particle_object_id.push_back(object_id);
            m_particle_masses.push_back(particle_mass);
            m_particle_volumes.push_back(particle_volume);
        }
    }

    virtual void LoadParticleBatchFromObjFile(const std::string &inputfile,
                                         float scale, glm::vec3 translate, glm::vec3 rotate,
                                         unsigned int particle_type, Real density, Real particle_volume,
                                         float radius, glm::vec3 color, glm::vec2 material)
    {
        m_object_types.push_back(particle_type);
        unsigned int object_id = (unsigned int)m_object_types.size() - 1;

        struct stat buffer;
        if (stat(inputfile.c_str(), &buffer) != 0)
        {
            std::cerr << ".obj file missed " << inputfile << std::endl;
        }
        
        std::vector<Particle> particles;
        glm::quat q(rotate);
        tinyobj::attrib_t attributes;
        std::vector<tinyobj::material_t> materials;
        std::vector<tinyobj::shape_t> shapes;
        std::string warning, error;
        if (!tinyobj::LoadObj(&attributes, &shapes, &materials, &warning, &error, inputfile.c_str()))
        {
            std::cout << warning << error << "\n";
        }
        unsigned int itTris = 0;
        for (const auto &shape : shapes)
        {
            for (const auto &index : shape.mesh.indices)
            {
                glm::vec3 __vertex(
                    attributes.vertices[3 * index.vertex_index],
                    attributes.vertices[3 * index.vertex_index + 1],
                    attributes.vertices[3 * index.vertex_index + 2]);
                // __vertex = glm::vec3(transform * glm::vec4(__vertex, 1.));
                __vertex = rotateVecQuat(q, __vertex);
                particles.push_back({scale * __vertex + translate, color});
            }
        }
        
        auto particle_batch = std::make_shared<ParticleBatch>(particles);
        particle_batch->AddRenderer(std::make_shared<SphereRenderer>(material, radius));
        this->AddParticleBatch(particle_batch);
        size_t num_particle = this->m_particle_batch_offsets.back().first->m_particles.size();
        Real particle_mass = density * particle_volume;
        for (size_t i = 0; i < num_particle; ++i)
        {
            m_particle_object_id.push_back(object_id);
            m_particle_masses.push_back(particle_mass);
            m_particle_volumes.push_back(particle_volume);
        }
    }
};

using Real = float;

void example0(std::shared_ptr<IQMPMScene<Real>> scene)
{
    scene->AddMPMCubeParticleBatch(glm::vec3(-6.0f, 12.0f, -5.0f), glm::vec3(6.0f, 18.0f, 5.0f), 0.08f,
                                   MPM_FLUID, 720000.0f,
                                   0.03f, glm::vec3(0.2f, 0.2f, 1.0f), glm::vec2(0.8f, 0.8f));
    scene->AddMPMCubeParticleBatch(glm::vec3(-6.0f, 6.0f, -1.0f), glm::vec3(5.0f, 8.0f, 1.0f), 0.08f,
                                   MPM_ELASTIC, 20000.0f,
                                   0.03f, glm::vec3(0.5f, 1.0f, 0.0f), glm::vec2(0.8f, 0.8f));
    scene->AddMPMCubeParticleBatch(glm::vec3(-1.0f, 3.0f, -6.0f), glm::vec3(1.0f, 5.0f, 5.0f), 0.08f,
                                   MPM_ELASTIC, 20000.0f,
                                   0.03f, glm::vec3(1.0f, 0.5f, 0.0f), glm::vec2(0.8f, 0.8f));
}

void example1(std::shared_ptr<IQMPMScene<Real>> scene)
{
    scene->AddMPMCubeParticleBatch(glm::vec3(-6.0f, 12.0f, -2.0f), glm::vec3(6.0f, 15.0f, 2.0f), 0.08f,
                                   MPM_FLUID, 144000.0f,
                                   0.03f, glm::vec3(0.2f, 0.2f, 1.0f), glm::vec2(0.8f, 0.8f));
    scene->AddMPMCubeParticleBatch(glm::vec3(-6.0f, 6.0f, -1.0f), glm::vec3(5.0f, 8.0f, 1.0f), 0.08f,
                                   MPM_ELASTIC, 20000.0f,
                                   0.03f, glm::vec3(0.5f, 1.0f, 0.0f), glm::vec2(0.8f, 0.8f));
    scene->AddMPMCubeParticleBatch(glm::vec3(-1.0f, 3.0f, -6.0f), glm::vec3(1.0f, 5.0f, 5.0f), 0.08f,
                                   MPM_ELASTIC, 20000.0f,
                                   0.03f, glm::vec3(1.0f, 0.5f, 0.0f), glm::vec2(0.8f, 0.8f));
}

void example2(std::shared_ptr<IQMPMScene<Real>> scene)
{
    std::vector<glm::vec3> colors = {
        glm::vec3(0.5f, 1.0f, 0.0f),
        glm::vec3(1.0f, 0.5f, 0.0f),
        glm::vec3(0.2f, 0.2f, 1.0f),
        glm::vec3(1.0f, 0.2f, 0.2f),
        glm::vec3(0.2f, 1.0f, 0.2f),
    };
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            for(int k = 0; k < 3; k++)
            {
                glm::vec3 lower_bound(-6.0f + i * 3.0f, 5.0f + j * 3.0f, -2.0f + k * 3.0f);
                glm::vec3 upper_bound(lower_bound.x + 2.0f, lower_bound.y + 2.0f, lower_bound.z + 2.0f);
                scene->AddMPMCubeParticleBatch(lower_bound, upper_bound, 0.08f,
                                               MPM_ELASTIC, 8000.0f,
                                               0.03f, colors[(i * 9 + j * 3 + k) % colors.size()], glm::vec2(0.8f, 0.8f));
            }
}

void example3(std::shared_ptr<IQMPMScene<Real>> scene)
{
    scene->AddMPMCubeParticleBatch(glm::vec3(-4.5f, 1.0f, -4.5f), glm::vec3(4.5f, 4.0f, 4.5f), 0.08f,
                                   MPM_FLUID, 432000.0f,
                                   0.03f, glm::vec3(0.2f, 0.2f, 1.0f), glm::vec2(0.8f, 0.8f));
    scene->AddMPMCubeParticleBatch(glm::vec3(-2.0f, 6.0f, -1.0f), glm::vec3(2.0f, 8.0f, 1.0f), 0.08f,
                                   MPM_ELASTIC, 12000.0f,
                                   0.03f, glm::vec3(0.5f, 1.0f, 0.0f), glm::vec2(0.8f, 0.8f));
}

void example4(std::shared_ptr<IQMPMScene<Real>> scene)
{
    scene->AddMPMCubeParticleBatch(glm::vec3(-4.5f, 1.0f, -4.5f), glm::vec3(4.5f, 4.0f, 4.5f), 0.08f,
                                   MPM_FLUID, 432000.0f,
                                   0.03f, glm::vec3(0.2f, 0.2f, 1.0f), glm::vec2(0.8f, 0.8f));
    scene->AddMPMCubeParticleBatch(glm::vec3(-2.0f, 6.0f, -1.0f), glm::vec3(2.0f, 8.0f, 1.0f), 0.08f,
                                   MPM_ELASTIC, 40000.0f,
                                   0.03f, glm::vec3(0.5f, 1.0f, 0.0f), glm::vec2(0.8f, 0.8f));
}

void example5(std::shared_ptr<IQMPMScene<Real>> scene)
{
    scene->LoadParticleBatchFromObjFile(std::string(ASSET_DIR) + "/AirplaneForFreeobj2.obj", 0.1f,
                                        glm::vec3(0.0f, 5.5f, 0.0f), glm::vec3(glm::radians(15.0f), 0.0f, 0.0f),
                                        MPM_ELASTIC, 1000.0f, 0.135f * 0.135f * 0.135f,
                                        0.03f, glm::vec3(0.5f, 0.5f, 0.1f), glm::vec2(0.8f, 0.8f));
    scene->AddMPMCubeParticleBatch(glm::vec3(-4.5f, 0.0f, -4.5f), glm::vec3(4.5f, 4.0f, 29.5f), 0.08f,
                                   MPM_FLUID, 1224000.0f,
                                   0.03f, glm::vec3(0.2f, 0.2f, 1.0f), glm::vec2(0.8f, 0.8f));
}

int main()
{
    auto app = GLFWApp::GetInstance("IQ-MPM Solver Example", 1600, 900);

    auto scene = std::make_shared<IQMPMScene<Real>>();
    scene->SetupScene();
    app->AddObject(scene);

    example5(scene);

    float dist = 0.2f;
    std::vector<float> bbox = {-5.0f, 0.0f, -5.0f, 5.0f, 15.0f, 30.0f};
    unsigned int boundary_thickness = 1;
    // app->AddObject(std::make_shared<CubeLineBox>(bbox, dist, glm::vec3(1.0f, 1.0f, 1.0f)));

    std::vector<Real> bbox_real;
    for (const auto &b : bbox)
        bbox_real.push_back(static_cast<Real>(b));
    auto solver = std::make_shared<IQMPMSolver<Real>>(scene->m_object_types, scene->m_particle_object_id,
                                                      scene->m_positions, scene->m_particle_masses, scene->m_particle_volumes,
                                                      bbox_real, (Real)dist, boundary_thickness);
    
    std::vector<Real> init_velocity;
    for(size_t i = 0; i < 337176u; ++i)
    {
        init_velocity.push_back(0.0f);
        init_velocity.push_back(-5.0f);
        init_velocity.push_back(25.0f);
    }
    solver->SetupInitVelocity(init_velocity);

    scene->SetSolver(solver);
    scene->SetStepPerFrame(2);
    scene->SetupConnectors();

    // unsigned int num_one_grid = solver->m_data.m_num_grid / solver->m_data.m_num_object;
    // auto field_vec_line = std::make_shared<LineSegment>(std::vector<LineSeg>(num_one_grid));
    // field_vec_line->AddRenderer(std::make_shared<SolidColorRenderer>(glm::vec3(1.0f, 0.0f, 0.0f)));
    // app->GetRenderSystem()->AddRenderObject(field_vec_line);
    // scene->AddConnector(std::make_shared<IQMPMDebugConnector<Real>>(field_vec_line, &solver->m_data, 1, dist * 0.5f));

    app->Run();

    return 0;
}

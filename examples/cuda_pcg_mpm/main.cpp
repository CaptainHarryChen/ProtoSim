#include <string>
#include <vector>
#include <memory>
#include <GLFWApp.h>
#include <Object/CubeLineBox.h>
#include <Scene/SimulationScene.h>
#include <Geometry/ParticleBatch.h>
#include <Mesh/MeshLoader.h>
#include <Solver/PCGMPMSolver.cuh>

template <typename Real>
class MPMScene : public SimulationScene<Real>
{
public:
    std::vector<unsigned int> m_particle_types;
    std::vector<Real> m_particle_masses;
    std::vector<Real> m_particle_volumes;

    virtual void AddMPMCubeParticleBatch(glm::vec3 lower_bound, glm::vec3 upper_bound, float dis,
                                         unsigned int particle_type, Real density,
                                         float radius, glm::vec3 color, glm::vec2 material)
    {
        SimulationScene<Real>::AddCubeParticleBatch(lower_bound, upper_bound, dis, radius, color, material);
        size_t num_particle = this->m_particle_batch_offsets.back().first->m_particles.size();
        Real total_volume = (upper_bound.x - lower_bound.x) * (upper_bound.y - lower_bound.y) * (upper_bound.z - lower_bound.z);
        Real particle_volume = dis * dis * dis;
        Real particle_mass = density * particle_volume;
        for (size_t i = 0; i < num_particle; ++i)
        {
            m_particle_types.push_back(particle_type);
            m_particle_masses.push_back(particle_mass);
            m_particle_volumes.push_back(particle_volume);
        }
    }
};

int main()
{
    using Real = float;

    auto app = GLFWApp::GetInstance("MPM Solver Example", 1600, 900);

    auto scene = std::make_shared<MPMScene<Real>>();
    scene->SetupScene();
    app->AddObject(scene);

    scene->AddMPMCubeParticleBatch(glm::vec3(-3.0f, 0.3f, -3.0f), glm::vec3(3.0f, 3.3f, 3.0f), 0.08f,
                                   MPM_FLUID, 1000.0f,
                                   0.03f, glm::vec3(0.2f, 0.2f, 1.0f), glm::vec2(0.8f, 0.8f));
    // scene->AddMPMCubeParticleBatch(glm::vec3(-6.0f, 6.0f, -1.0f), glm::vec3(5.0f, 8.0f, 1.0f), 0.08f,
    //                                MPM_ELASTIC, 500.0f,
    //                                0.03f, glm::vec3(0.5f, 1.0f, 0.0f), glm::vec2(0.8f, 0.8f));
    // scene->AddMPMCubeParticleBatch(glm::vec3(-1.0f, 3.0f, -6.0f), glm::vec3(1.0f, 5.0f, 5.0f), 0.08f,
    //                                MPM_ELASTIC, 500.0f,
    //                                0.03f, glm::vec3(1.0f, 0.5f, 0.0f), glm::vec2(0.8f, 0.8f));

    float dist = 0.2f;
    // std::vector<float> bbox = {-10.0f, 0.0f, -10.0f, 10.0f, 20.0f, 10.0f};
    std::vector<float> bbox = {-5.0f, 0.0f, -5.0f, 5.0f, 10.0f, 5.0f};
    unsigned int boundary_thickness = 1;
    app->AddObject(std::make_shared<CubeLineBox>(bbox, dist, glm::vec3(1.0f, 1.0f, 1.0f)));

    auto solver = std::make_shared<PCGMPMSolver<Real>>(scene->m_positions, scene->m_particle_types, scene->m_particle_masses, scene->m_particle_volumes,
                                                    bbox, dist, boundary_thickness);
    // solver->m_verbose = true;
    scene->SetSolver(solver);
    // scene->SetStepPerFrame(2);
    scene->SetupConnectors();

    app->Run();

    return 0;
}

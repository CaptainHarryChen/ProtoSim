#include <string>
#include <vector>
#include <memory>
#include <GLFWApp.h>
#include <Object/CubeLineBox.h>
#include <Scene/SimulationScene.h>
#include <Geometry/ParticleBatch.h>
#include <Mesh/MeshLoader.h>
#include <Solver/MPMSolver.cuh>

template <typename Real>
class MPMScene : public SimulationScene<Real>
{
public:
    std::vector<unsigned int> m_particle_types;
    std::vector<Real> m_particle_masses;
    std::vector<Real> m_particle_volumes;

    virtual void AddMPMCubeParticleBatch(glm::vec3 lower_bound, glm::vec3 upper_bound, float dis,
                                         unsigned int particle_type, Real total_mass,
                                         float radius, glm::vec3 color, glm::vec2 material)
    {
        SimulationScene<Real>::AddCubeParticleBatch(lower_bound, upper_bound, dis, radius, color, material);
        size_t num_particle = this->m_particle_batch_offsets.back().first->m_particles.size();
        Real total_volume = (upper_bound.x - lower_bound.x) * (upper_bound.y - lower_bound.y) * (upper_bound.z - lower_bound.z);
        Real particle_volume = dis * dis * dis;
        Real particle_mass = total_mass * particle_volume / total_volume;
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

    scene->AddMPMCubeParticleBatch(glm::vec3(-1.0f, 7.0f, -6.0f), glm::vec3(1.0f, 9.0f, 5.0f), 0.15f,
                                   MPM_ELASTIC, 22000.0f,
                                   0.05f, glm::vec3(0.5f, 1.0f, 0.0f), glm::vec2(0.8f, 0.8f));
    scene->AddMPMCubeParticleBatch(glm::vec3(-1.0f, 3.0f, -6.0f), glm::vec3(1.0f, 5.0f, 5.0f), 0.15f,
                                   MPM_FLUID, 22000.0f,
                                   0.05f, glm::vec3(0.0f, 1.0f, 0.5f), glm::vec2(0.8f, 0.8f));

    float dist = 0.4f;
    std::vector<float> bbox = {-10.0f, 0.0f, -12.0f, 10.0f, 20.0f, 8.0f};
    unsigned int boundary_thickness = 1;
    app->AddObject(std::make_shared<CubeLineBox>(bbox, dist, glm::vec3(1.0f, 1.0f, 1.0f)));

    auto solver = std::make_shared<MPMSolver<Real>>(scene->m_positions, scene->m_particle_types, scene->m_particle_masses, scene->m_particle_volumes,
                                                    bbox, dist, boundary_thickness);
    scene->SetSolver(solver);
    scene->SetupConnectors();

    app->Run();

    return 0;
}

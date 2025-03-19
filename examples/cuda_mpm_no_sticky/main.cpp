#include <string>
#include <vector>
#include <memory>
#include <GLFWApp.h>
#include <Object/CubeLineBox.h>
#include <Scene/SimulationScene.h>
#include <Geometry/ParticleBatch.h>
#include <Mesh/MeshLoader.h>
#include <Solver/IQMPMSolver.cuh>
#include <Geometry/LineSegment.h>
#include <Render/RenderSystem.h>
#include <Mesh/SolidColorRenderer.h>
#include "IQMPMDebugConnector.cuh"

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
};

int main()
{
    using Real = float;

    auto app = GLFWApp::GetInstance("IQ-MPM Solver Example", 1600, 900);

    auto scene = std::make_shared<IQMPMScene<Real>>();
    scene->SetupScene();
    app->AddObject(scene);

    scene->AddMPMCubeParticleBatch(glm::vec3(-6.0f, 12.0f, -5.0f), glm::vec3(6.0f, 18.0f, 5.0f), 0.08f,
                                   MPM_FLUID, 720000.0f,
                                   0.03f, glm::vec3(0.2f, 0.2f, 1.0f), glm::vec2(0.8f, 0.8f));
    scene->AddMPMCubeParticleBatch(glm::vec3(-6.0f, 6.0f, -1.0f), glm::vec3(5.0f, 8.0f, 1.0f), 0.08f,
                                   MPM_ELASTIC, 20000.0f,
                                   0.03f, glm::vec3(0.5f, 1.0f, 0.0f), glm::vec2(0.8f, 0.8f));
    scene->AddMPMCubeParticleBatch(glm::vec3(-1.0f, 3.0f, -6.0f), glm::vec3(1.0f, 5.0f, 5.0f), 0.08f,
                                   MPM_ELASTIC, 20000.0f,
                                   0.03f, glm::vec3(1.0f, 0.5f, 0.0f), glm::vec2(0.8f, 0.8f));

    float dist = 0.2f;
    std::vector<float> bbox = {-10.0f, 0.0f, -12.0f, 10.0f, 20.0f, 8.0f};
    unsigned int boundary_thickness = 1;
    app->AddObject(std::make_shared<CubeLineBox>(bbox, dist, glm::vec3(1.0f, 1.0f, 1.0f)));

    auto solver = std::make_shared<IQMPMSolver<Real>>(scene->m_object_types, scene->m_particle_object_id,
                                                      scene->m_positions, scene->m_particle_masses, scene->m_particle_volumes,
                                                      bbox, dist, boundary_thickness);
    scene->SetSolver(solver);
    scene->SetStepPerFrame(2);
    scene->SetupConnectors();

    unsigned int num_one_grid = solver->m_data.m_num_grid / solver->m_data.m_num_object;
    auto field_vec_line = std::make_shared<LineSegment>(std::vector<LineSeg>(num_one_grid));
    field_vec_line->AddRenderer(std::make_shared<SolidColorRenderer>(glm::vec3(1.0f, 0.0f, 0.0f)));
    app->GetRenderSystem()->AddRenderObject(field_vec_line);
    scene->AddConnector(std::make_shared<IQMPMDebugConnector<Real>>(field_vec_line, &solver->m_data, 1, dist * 0.5f));

    app->Run();

    return 0;
}

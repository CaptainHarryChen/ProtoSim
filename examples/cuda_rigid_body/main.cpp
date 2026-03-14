#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <common.h>
#include <Solver/RigidSolver.cuh>
#include <viewer/utils/PrimitiveGenerator.h>
#include <viewer/Renderer/PbrRenderer.h>
#include <viewer/Renderer/SphereRenderer.h>
#include <viewer/Renderer/CapsuleRenderer.h>
#include <viewer/RenderObject/ParticleBatch.h>
#include <viewer/RenderObject/SingleCapsule.h>
#include <viewer/Connector/ModelConnector.cuh>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <cstdlib>

template <typename Real>
class RigidBodyScene : public SimulationScene<Real>
{
public:
    std::vector<Real> m_positions;
    std::vector<Real> m_orientations;
    std::vector<Real> m_linear_velocities;
    std::vector<Real> m_angular_velocities;
    std::vector<Real> m_masses;
    std::vector<int> m_shapes;
    std::vector<Real> m_shape_params;

    std::vector<std::shared_ptr<viewer::RenderObject>> m_rigid_objects;

    bool m_enableHighlightRolling = false;

    void SetHighlightRolling(bool enable) { m_enableHighlightRolling = enable; }

    void AddBox(glm::vec3 position, glm::quat orientation,
                glm::vec3 half_extents, Real mass,
                std::vector<glm::vec3> render_material)
    {
        m_positions.push_back(position.x);
        m_positions.push_back(position.y);
        m_positions.push_back(position.z);
        m_orientations.push_back(orientation.w);
        m_orientations.push_back(orientation.x);
        m_orientations.push_back(orientation.y);
        m_orientations.push_back(orientation.z);
        m_linear_velocities.push_back(0);
        m_linear_velocities.push_back(0);
        m_linear_velocities.push_back(0);
        m_angular_velocities.push_back(0);
        m_angular_velocities.push_back(0);
        m_angular_velocities.push_back(0);
        m_masses.push_back(mass);
        m_shapes.push_back(RIGID_BODY_BOX);
        m_shape_params.push_back(half_extents.x);
        m_shape_params.push_back(half_extents.y);
        m_shape_params.push_back(half_extents.z);

        auto mesh = viewer::PrimitiveGenerator::GenerateBox(half_extents);
        auto pbrRenderer = std::make_shared<viewer::PbrRenderer>(render_material);
        pbrRenderer->SetEnableShadow(false);
        mesh->AddRenderer(pbrRenderer);
        mesh->m_model_mat = glm::translate(glm::mat4(1.0f), position) * glm::toMat4(orientation);
        m_rigid_objects.push_back(mesh);
        viewer::GLFWApp::GetInstance()->GetRenderSystem()->AddRenderObject(mesh);
    }

    void AddSphere(glm::vec3 position, glm::quat orientation,
                   Real radius, Real mass,
                   std::vector<glm::vec3> render_material)
    {
        m_positions.push_back(position.x);
        m_positions.push_back(position.y);
        m_positions.push_back(position.z);
        m_orientations.push_back(orientation.w);
        m_orientations.push_back(orientation.x);
        m_orientations.push_back(orientation.y);
        m_orientations.push_back(orientation.z);
        m_linear_velocities.push_back(0);
        m_linear_velocities.push_back(0);
        m_linear_velocities.push_back(0);
        m_angular_velocities.push_back(0);
        m_angular_velocities.push_back(0);
        m_angular_velocities.push_back(0);
        m_masses.push_back(mass);
        m_shapes.push_back(RIGID_BODY_SPHERE);
        m_shape_params.push_back(radius);
        m_shape_params.push_back(0);
        m_shape_params.push_back(0);

        std::vector<viewer::Particle> particles = {
            {glm::vec3(0.0f, 0.0f, 0.0f), render_material[0]}};
        auto batch = std::make_shared<viewer::ParticleBatch>(particles);
        auto sphereRenderer = std::make_shared<viewer::SphereRenderer>(
            glm::vec2(render_material[1].x, render_material[1].y), radius, m_enableHighlightRolling);
        batch->AddRenderer(sphereRenderer);
        batch->m_model_mat = glm::translate(glm::mat4(1.0f), position) * glm::toMat4(orientation);
        m_rigid_objects.push_back(batch);
        viewer::GLFWApp::GetInstance()->GetRenderSystem()->AddRenderObject(batch);
    }

    void AddCapsule(glm::vec3 position, glm::quat orientation,
                    Real radius, Real half_height, Real mass,
                    std::vector<glm::vec3> render_material)
    {
        m_positions.push_back(position.x);
        m_positions.push_back(position.y);
        m_positions.push_back(position.z);
        m_orientations.push_back(orientation.w);
        m_orientations.push_back(orientation.x);
        m_orientations.push_back(orientation.y);
        m_orientations.push_back(orientation.z);
        m_linear_velocities.push_back(0);
        m_linear_velocities.push_back(0);
        m_linear_velocities.push_back(0);
        m_angular_velocities.push_back(0);
        m_angular_velocities.push_back(0);
        m_angular_velocities.push_back(0);
        m_masses.push_back(mass);
        m_shapes.push_back(RIGID_BODY_CAPSULE);
        m_shape_params.push_back(radius);
        m_shape_params.push_back(half_height);
        m_shape_params.push_back(0);

        auto capsule = std::make_shared<viewer::SingleCapsule>(render_material[0]);
        auto capsuleRenderer = std::make_shared<viewer::CapsuleRenderer>(
            glm::vec2(render_material[1].x, render_material[1].y), radius, half_height, m_enableHighlightRolling);
        capsule->AddRenderer(capsuleRenderer);
        capsule->m_model_mat = glm::translate(glm::mat4(1.0f), position) * glm::toMat4(orientation);
        m_rigid_objects.push_back(capsule);
        viewer::GLFWApp::GetInstance()->GetRenderSystem()->AddRenderObject(capsule);
    }

    void SortByShape()
    {
        size_t num_bodies = m_masses.size();
        if (num_bodies == 0)
            return;

        std::vector<size_t> indices(num_bodies);
        for (size_t i = 0; i < num_bodies; ++i)
            indices[i] = i;

        std::stable_sort(indices.begin(), indices.end(), [this](size_t a, size_t b)
                         { return m_shapes[a] < m_shapes[b]; });

        std::vector<Real> new_positions(num_bodies * 3);
        std::vector<Real> new_orientations(num_bodies * 4);
        std::vector<Real> new_linear_velocities(num_bodies * 3);
        std::vector<Real> new_angular_velocities(num_bodies * 3);
        std::vector<Real> new_masses(num_bodies);
        std::vector<int> new_shapes(num_bodies);
        std::vector<Real> new_shape_params(num_bodies * 3);
        std::vector<std::shared_ptr<viewer::RenderObject>> new_objects(num_bodies);

        for (size_t i = 0; i < num_bodies; ++i)
        {
            size_t src = indices[i];
            for (int j = 0; j < 3; ++j)
            {
                new_positions[i * 3 + j] = m_positions[src * 3 + j];
                new_linear_velocities[i * 3 + j] = m_linear_velocities[src * 3 + j];
                new_angular_velocities[i * 3 + j] = m_angular_velocities[src * 3 + j];
                new_shape_params[i * 3 + j] = m_shape_params[src * 3 + j];
            }
            for (int j = 0; j < 4; ++j)
            {
                new_orientations[i * 4 + j] = m_orientations[src * 4 + j];
            }
            new_masses[i] = m_masses[src];
            new_shapes[i] = m_shapes[src];
            new_objects[i] = m_rigid_objects[src];
        }

        m_positions = std::move(new_positions);
        m_orientations = std::move(new_orientations);
        m_linear_velocities = std::move(new_linear_velocities);
        m_angular_velocities = std::move(new_angular_velocities);
        m_masses = std::move(new_masses);
        m_shapes = std::move(new_shapes);
        m_shape_params = std::move(new_shape_params);
        m_rigid_objects = std::move(new_objects);
    }
};

template <typename Real>
void GenerateRandomScene(std::shared_ptr<RigidBodyScene<Real>> scene)
{
    const int gridSize = 5;
    const float cellSize = 1.5f;
    const float startHeight = 5.0f;
    srand(12);

    for (int i = 0; i < gridSize; ++i)
    {
        for (int j = 0; j < gridSize; ++j)
        {
            for (int k = 0; k < gridSize; ++k)
            {
                float x = (i - gridSize / 2.0f + 0.5f) * cellSize;
                float y = startHeight + j * cellSize;
                float z = (k - gridSize / 2.0f + 0.5f) * cellSize;
                glm::vec3 position(x, y, z);

                int shapeType = rand() % 1;
                float maxDim = cellSize * 0.4f;
                float randomScale = 0.2f + (rand() % 50) / 100.0f * maxDim;
                randomScale = std::min(randomScale, maxDim);
                float mass = 0.5f + (rand() % 15) / 10.0f;

                float angle = (rand() % 360) * glm::pi<Real>() / 180.0f;
                glm::vec3 axis(normalize(glm::vec3(
                    static_cast<Real>(rand()) / RAND_MAX - 0.5f,
                    static_cast<Real>(rand()) / RAND_MAX - 0.5f,
                    static_cast<Real>(rand()) / RAND_MAX - 0.5f)));
                glm::quat rotation = glm::angleAxis(angle, axis);

                glm::vec3 color;
                if (shapeType == 0)
                {
                    color = glm::vec3(0.8f, 0.3f, 0.3f);
                    scene->AddBox(
                        position,
                        rotation,
                        glm::vec3(randomScale, randomScale * 0.8f, randomScale * 1.2f),
                        mass,
                        {color, glm::vec3(0.1f, 0.5f, 0.1f)});
                }
                else if (shapeType == 1)
                {
                    color = glm::vec3(0.3f, 0.3f, 0.8f);
                    scene->AddSphere(
                        position,
                        rotation,
                        randomScale,
                        mass,
                        {color, glm::vec3(0.1f, 0.3f, 0.1f)});
                }
                else
                {
                    color = glm::vec3(0.3f, 0.8f, 0.3f);
                    scene->AddCapsule(
                        position,
                        rotation,
                        randomScale * 0.5f,
                        randomScale,
                        mass,
                        {color, glm::vec3(0.1f, 0.5f, 0.1f)});
                }
            }
        }
    }
}

int main()
{
    using Real = float;

    auto app = viewer::GLFWApp::GetInstance("Rigid Body Solver Example", 1600, 900);

    auto scene = std::make_shared<RigidBodyScene<Real>>();
    scene->SetHighlightRolling(true);
    scene->SetupScene();
    app->AddObject(scene);

    GenerateRandomScene(scene);
    // scene->AddBox(
    //     glm::vec3(0.0f, 2.0f, 0.0f),
    //     glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
    //     glm::vec3(0.5f, 0.5f, 0.5f),
    //     1.0f,
    //     {glm::vec3(0.3f, 0.8f, 0.3f), glm::vec3(0.1f, 0.5f, 0.1f)});
    // scene->AddCapsule(
    //     glm::vec3(0.6f, 4.0f, 0.0f),
    //     glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
    //     0.3f,
    //     0.8f,
    //     0.5f,
    //     {glm::vec3(0.3f, 0.8f, 0.3f), glm::vec3(0.1f, 0.5f, 0.1f)});

    scene->SortByShape();

    auto solver = std::make_shared<RigidSolver<Real>>(
        scene->m_positions,
        scene->m_orientations,
        scene->m_linear_velocities,
        scene->m_angular_velocities,
        scene->m_masses,
        scene->m_shapes,
        scene->m_shape_params);
    scene->SetSolver(solver);

    auto model_connector = std::make_shared<viewer::ModelConnector<Real>>(
        scene->m_rigid_objects,
        solver->GetDevicePositions(),
        solver->GetDeviceOrientations(),
        static_cast<unsigned int>(scene->m_masses.size()));
    scene->AddConnector(model_connector);

    app->Run();

    return 0;
}

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <common.h>
#include <Solver/RigidSolver.cuh>
#include <viewer/utils/PrimitiveGenerator.h>
#include <viewer/Renderer/PbrRenderer.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

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

    std::vector<std::shared_ptr<viewer::Mesh>> m_rigid_meshes;

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
        mesh->AddRenderer(std::make_shared<viewer::PbrRenderer>(render_material));
        mesh->m_model_mat = glm::translate(glm::mat4(1.0f), position) * glm::toMat4(orientation);
        m_rigid_meshes.push_back(mesh);
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

        auto mesh = viewer::PrimitiveGenerator::GenerateSphere(radius);
        mesh->AddRenderer(std::make_shared<viewer::PbrRenderer>(render_material));
        mesh->m_model_mat = glm::translate(glm::mat4(1.0f), position) * glm::toMat4(orientation);
        m_rigid_meshes.push_back(mesh);
        viewer::GLFWApp::GetInstance()->GetRenderSystem()->AddRenderObject(mesh);
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

        auto mesh = viewer::PrimitiveGenerator::GenerateCapsule(radius, half_height);
        mesh->AddRenderer(std::make_shared<viewer::PbrRenderer>(render_material));
        mesh->m_model_mat = glm::translate(glm::mat4(1.0f), position) * glm::toMat4(orientation);
        m_rigid_meshes.push_back(mesh);
        viewer::GLFWApp::GetInstance()->GetRenderSystem()->AddRenderObject(mesh);
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
        std::vector<std::shared_ptr<viewer::Mesh>> new_meshes(num_bodies);

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
            new_meshes[i] = m_rigid_meshes[src];
        }

        m_positions = std::move(new_positions);
        m_orientations = std::move(new_orientations);
        m_linear_velocities = std::move(new_linear_velocities);
        m_angular_velocities = std::move(new_angular_velocities);
        m_masses = std::move(new_masses);
        m_shapes = std::move(new_shapes);
        m_shape_params = std::move(new_shape_params);
        m_rigid_meshes = std::move(new_meshes);
    }
};

int main()
{
    using Real = float;

    auto app = viewer::GLFWApp::GetInstance("Rigid Body Solver Example", 1600, 900);

    auto scene = std::make_shared<RigidBodyScene<Real>>();
    scene->SetupScene();
    app->AddObject(scene);

    scene->AddBox(
        glm::vec3(0.0f, 2.0f, 0.0f),
        glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        glm::vec3(0.5f, 0.5f, 0.5f),
        1.0f,
        {glm::vec3(0.8f, 0.2f, 0.2f), glm::vec3(0.1f, 0.5f, 0.1f)});

    scene->AddBox(
        glm::vec3(1.5f, 3.0f, 0.0f),
        glm::angleAxis(glm::radians(30.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
        glm::vec3(0.3f, 0.6f, 0.3f),
        2.0f,
        {glm::vec3(0.2f, 0.8f, 0.2f), glm::vec3(0.1f, 0.5f, 0.1f)});

    scene->AddSphere(
        glm::vec3(-1.5f, 2.0f, 0.0f),
        glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        0.5f,
        1.0f,
        {glm::vec3(0.2f, 0.2f, 0.8f), glm::vec3(0.1f, 0.3f, 0.1f)});

    scene->AddSphere(
        glm::vec3(-1.5f, 4.0f, 1.0f),
        glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        0.3f,
        0.5f,
        {glm::vec3(0.8f, 0.8f, 0.2f), glm::vec3(0.1f, 0.3f, 0.1f)});

    scene->AddCapsule(
        glm::vec3(0.0f, 4.0f, 1.5f),
        glm::angleAxis(glm::radians(45.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        0.25f,
        0.5f,
        1.5f,
        {glm::vec3(0.8f, 0.4f, 0.8f), glm::vec3(0.1f, 0.5f, 0.1f)});

    scene->AddCapsule(
        glm::vec3(2.0f, 5.0f, -1.0f),
        glm::angleAxis(glm::radians(60.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
        0.2f,
        0.4f,
        1.0f,
        {glm::vec3(0.4f, 0.8f, 0.8f), glm::vec3(0.1f, 0.5f, 0.1f)});

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

    app->Run();

    return 0;
}

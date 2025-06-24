#include <string>
#include <vector>
#include <memory>
#include <igl/random_points_on_mesh.h>
#include <Eigen/Core>
#include <GLFWApp.h>
#include <Scene/SimulationScene.h>
#include <Loader/TetrahedronLoader.h>
#include <Geometry/ParticleBatch.h>
#include <Geometry/SphereRenderer.h>
#include <Render/RenderSystem.h>
#include <Mesh/Mesh.h>
#include <Mesh/PbrRenderer.h>
#include <Solver/ProjectiveDynamicsSolver.cuh>
#include <proj_config.h>

template <typename Real>
class TetrahedronScene : public SimulationScene<Real>
{
public:
    // std::vector<Real> m_positions;
    std::vector<unsigned int> m_tetrahedras;
    std::vector<unsigned int> m_object_tetrahedras_offsets;
    std::vector<Real> m_tetrahedras_densities;

    void LoadTetrahedron(std::string inputfile, Real density,
                         float scale, glm::vec3 translate, glm::vec3 rotate,
                         std::vector<glm::vec3> render_material)
    {
        std::vector<float> positions;
        std::vector<unsigned int> surface_triangles;
        std::vector<unsigned int> tetrahedras;
        TetrahedronLoader::LoadTetrahedron(inputfile, scale, translate, rotate, positions, surface_triangles, tetrahedras);

        std::vector<Vertex> vertices(positions.size() / 3);
        for (size_t i = 0; i < positions.size() / 3; ++i)
        {
            vertices[i].position = glm::vec3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
            vertices[i].normal = glm::vec3(0.0f, 0.0f, 0.0f);
            vertices[i].tex_coords = glm::vec2(0.0f, 0.0f);
        }
        auto mesh = std::make_shared<Mesh>(vertices, surface_triangles);
        mesh->AddRenderer(std::make_shared<PbrRenderer>(render_material));
        SimulationScene<Real>::AddMesh(mesh);

        unsigned int offset = (unsigned int)SimulationScene<Real>::m_mesh_offsets.back().second / 3;
        m_object_tetrahedras_offsets.push_back(offset * 4);
        m_tetrahedras.resize(m_tetrahedras.size() + tetrahedras.size());
        for (size_t i = 0; i < tetrahedras.size(); ++i)
            m_tetrahedras[i + offset * 4] = tetrahedras[i] + offset;
        m_tetrahedras_densities.resize(m_tetrahedras.size() / 4);
        for (size_t i = 0; i < m_tetrahedras_densities.size(); ++i)
            m_tetrahedras_densities[i] = density;

        Eigen::MatrixXd V(positions.size() / 3, 3);
        for (size_t i = 0; i < positions.size() / 3; ++i)
        {
            V(i, 0) = positions[i * 3 + 0];
            V(i, 1) = positions[i * 3 + 1];
            V(i, 2) = positions[i * 3 + 2];
        }

        Eigen::MatrixXi F(surface_triangles.size() / 3, 3);
        for (size_t i = 0; i < surface_triangles.size() / 3; ++i)
        {
            F(i, 0) = surface_triangles[i * 3 + 0];
            F(i, 1) = surface_triangles[i * 3 + 1];
            F(i, 2) = surface_triangles[i * 3 + 2];
        }

        // 需要生成的采样点数
        int num_samples = 500;

        // 输出变量
        Eigen::MatrixXd samples;      // 采样点的3D坐标
        Eigen::MatrixXd bary_coords;  // 采样点的重心坐标
        Eigen::VectorXi face_indices; // 每个点所在面片的索引

        // 执行表面采样
        igl::random_points_on_mesh(
            num_samples, // 采样点数
            V, F,        // 输入网格
            bary_coords, // 输出：重心坐标 (num_samples x 3)
            face_indices, // 输出：面片索引 (num_samples x 1)
            samples     // 输出：采样点坐标 (num_samples x 3)
        );
        
        std::vector<Particle> particles(samples.rows());
        for (int i = 0; i < samples.rows(); ++i)
        {
            particles[i].Position = glm::vec3(samples(i, 0), samples(i, 1), samples(i, 2));
            particles[i].Color = glm::vec3(1.0f, 0.0f, 0.0f);
        }
        auto particle_batch = std::make_shared<ParticleBatch>(particles);
        particle_batch->AddRenderer(std::make_shared<SphereRenderer>(glm::vec2(0.1f), 0.02f));
        GLFWApp::GetInstance()->GetRenderSystem()->AddRenderObject(particle_batch);
    }
};

int main()
{
    using Real = float;

    auto app = GLFWApp::GetInstance("Particle Sample Example", 1600, 900);

    auto scene = std::make_shared<TetrahedronScene<Real>>();
    scene->SetupScene();
    app->AddObject(scene);

    scene->LoadTetrahedron(std::string(ASSET_DIR) + "/bunny", 1000.0f,
                           10.0f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(glm::radians(-45.0f), 0.0f, 0.0f),
                           {glm::vec3(0.5f, 0.5f, 1.0f), glm::vec3(0.1f, 0.1f, 0.1f)});

    app->Run();

    return 0;
}

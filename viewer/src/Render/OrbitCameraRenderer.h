#pragma once
#include <memory>
#include <vector>
#include <Render/RenderSystem.h>
#include <Render/OrbitControl.h>

class CubeLight;
class SurfaceMesh;
class Sphere;
class LineSegment;

// TODO: Split the control and rendering. Split the camera and the renderer.
class OrbitCameraRenderer : public RenderSystem
{
public:
    static std::shared_ptr<OrbitCameraRenderer> GetInstance();

    OrbitCameraRenderer(std::string name = "Viewer", int init_width = 1600, int init_height = 900);
    virtual ~OrbitCameraRenderer();

    glm::vec3 clearColor = {0.5, 0.5, 1.0};

    virtual bool ProcessControl() override;
    virtual void RenderOneFrame() override;

protected:
    GLFWwindow *window;
    std::shared_ptr<OrbitControl> camera;

    // lights
    std::vector<std::shared_ptr<CubeLight>> cubelights;

    // objects for every shaders
    std::vector<std::shared_ptr<SurfaceMesh>> meshes;
    std::vector<std::shared_ptr<Sphere>> spheres;
    std::vector<std::shared_ptr<LineSegment>> lines;
};

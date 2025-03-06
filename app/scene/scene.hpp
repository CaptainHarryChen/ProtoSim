#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <iostream>
#include "viewer_config.h"
#include "Shader.h"
#include "stb_image.h"
#include "QuatCamera.h"
#include "MeshBase.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <ShadowMapping.h>
#include <Sphere.h>
#include <LineSegment.h>
#include "Accessories.h"
const unsigned int nLights = 4;

class scene
{
public:
    scene(std::string _name = "Scene", bool _isShadowMappingCubic = false);
    ~scene();
    void lightingSetUp();
    void updateLighting();
    void drawLights();
    void drawMeshes();
    void drawLines();
    void setLines(LineSegment lines, glm::vec3 color);
    void drawParticles();
    void loadStaticParticles(glm::vec3 lower_bound, glm::vec3 upper_bound, float delta,
        float radius = 0.05, glm::vec3 color = glm::vec3(1.f), glm::vec2 material = glm::vec2(.1));
    void loadSurfaceMeshFromOBJ(std::string inputfile, float scale,
		glm::vec3 translate, glm::vec3 rotate, // pitch, yaw, roll
		std::vector<glm::vec3> material);
    void loadSurfaceMeshFromOBJ(std::string inputfile, float scale,
      glm::vec3 translate, glm::vec3 rotate, // pitch, yaw, roll
      std::string albedo = std::string(VIEWER_DIR) + "/data/default_texture/albedo.png",
      std::string metallic = std::string(VIEWER_DIR) + "/data/default_texture/metallic.png",
      std::string normal = std::string(VIEWER_DIR) + "/data/default_texture/normal.png",
      std::string roughness = std::string(VIEWER_DIR) + "/data/default_texture/roughness.png",
      std::string ao = std::string(VIEWER_DIR) + "/data/default_texture/ao.png");
    void run();
    static bool isShadowMappingCubic; // 0: perspective shadow mapping, 1: shadow cube mapping
    bool showEdge = false;
    bool fakeFloor = true;

protected:
    GLFWwindow* window;
    std::shared_ptr<QuatCamera> camera;
    std::shared_ptr<Floor> floor;

    std::string name = "Scene";
    glm::vec3 clearColor = glm::vec3(.5, .5, 1.);

    float IG_lightPoses[nLights][3] = { -3.0f, 3.0f, -3.0f, -3.0f, 3.0f, 3.0f, 3.0f, 3.0f, -3.0f, 3.0f, 3.0f, 3.0f };
    float IG_lightColors[nLights][3] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    std::vector<Light> lights;
    bool IG_lightsOn[nLights] = { 0, 0, 0, 1 };
    bool IG_hideLights[nLights] = { 0, 0, 0, 0 };
    bool shadowMapping = true;
    int width = 1600, height = 900;
    std::vector<std::shared_ptr<SurfaceMesh>> meshes;
    std::vector<std::shared_ptr<Spheres>> particles;
    std::vector<glm::mat4> lightSpaceMatrices;
    glm::mat4 model, view, projection;
    std::vector<glm::vec3> lightPoses;
    std::vector<glm::vec3> lightColors;
    std::vector<unsigned int> depthMapIDs;
    std::vector<bool> lightsOn = { IG_lightsOn[0], IG_lightsOn[1], IG_lightsOn[2], IG_lightsOn[3] };
    std::vector<MeshLoader> static_mesh_loaders;
    std::vector<std::shared_ptr<LineSegment>> static_lines; glm::vec3 line_color = glm::vec3(0.);
    void setImGUI(float fps = 0);

};
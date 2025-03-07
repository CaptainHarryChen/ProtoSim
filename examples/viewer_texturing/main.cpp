#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <iostream>
#include <Render/Shader.h>
#include <Render/OrbitControl.h>
#include <stb_image.h>
#include <Mesh/MeshLoader.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <Geometry/ShadowMapping.h>
#include <Geometry/Sphere.h>
#include <Geometry/LineSegment.h>
#include <Object/CubeLight.h>
#include <Object/Floor.h>

int width = 1600;
int height = 900;
std::string name = "Viewer";
glm::vec3 clearColor(.5, .5, 1.);

int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);
#endif

    GLFWwindow* window = glfwCreateWindow(width, height, name.c_str(), nullptr, nullptr);
    OrbitControl camera(window);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, OrbitControl::framebuffer_size_callback);
    glfwSetScrollCallback(window, OrbitControl::scroll_callback);
    glfwSetMouseButtonCallback(window, OrbitControl::mousebutton_callback);
    glfwSetCursorPosCallback(window, OrbitControl::cursor_callback);

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    //glfwSwapInterval(1);
    Floor floor(100.0f, false);
    MeshLoader bunny1(std::string(VIEWER_DIR) + "/data/objs/venus.obj", 0.01f, glm::vec3(-1.0, 0.0, 0.0), glm::vec3(glm::radians(-90.0f), 0., 0.),
        { glm::vec3(1.), glm::vec3(0.1, 0.1, 0.1) }, false);
    MeshLoader bunny2(std::string(VIEWER_DIR) + std::string("/data/objs/CartoonSquirelModel/CartoonSquirelModel.obj"), 0.5, glm::vec3(0.,0., -1.), glm::vec3(0.), false,
        std::string(VIEWER_DIR) + std::string("/data/objs/CartoonSquirelModel/DiffSqurel.tga"));
    MeshLoader bunny3(std::string(VIEWER_DIR) + "/data/objs/dragon.obj", 6.0, glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.),
        { glm::vec3(0., 0., 1.), glm::vec3(0.1, 0.1, 0.1) }, false);

    std::vector<std::shared_ptr<SurfaceMesh>> meshes;
    meshes.push_back(floor.getMesh());
    meshes.push_back(bunny1.getMesh());
    meshes.push_back(bunny2.getMesh());
    meshes.push_back(bunny3.getMesh());

    const int nLights = 4;

    float IG_lightPoses[nLights][3] = {-3.0f, 3.0f, -3.0f, -3.0f, 3.0f, 3.0f, 3.0f, 3.0f, -3.0f, 3.0f, 3.0f, 3.0f};
    float IG_lightColors[nLights][3] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    std::vector<CubeLight> lights;
    lights.push_back(CubeLight(glm::vec3(-3.0f, 3.0f, -3.0f), glm::vec3(1.0, 0.0, 0.0), false));
    lights.push_back(CubeLight(glm::vec3(-3.0f, 3.0f, 3.0f), glm::vec3(0.0, 1.0, 0.0), false));
    lights.push_back(CubeLight(glm::vec3(3.0f, 3.0f, -3.0f), glm::vec3(0.0, 0.0, 1.0), false));
    lights.push_back(CubeLight(glm::vec3(3.0f, 3.0f, 3.0f), glm::vec3(1.0, 1.0, 1.0), false));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    
    bool IG_lightsOn[] = {1, 0, 0, 1};
    bool IG_hideLights[] = { 0, 0, 0, 0 };
    bool shadowMapping = true;
    std::vector<Point> points;
    points.push_back({glm::vec3(2.), glm::vec3(1., 1., 0.)});
    Sphere spheres(points, glm::vec2(0., 1.));
    std::vector<glm::vec3> lineVerts;
    lineVerts.push_back(glm::vec3(-1., 0.0, -1.));
    lineVerts.push_back(glm::vec3(-1., 0.0,  1.));
    lineVerts.push_back(glm::vec3( 1., 0.0, -1.));
    lineVerts.push_back(glm::vec3( 1., 0.0,  1.));
    lineVerts.push_back(glm::vec3(-1., 2.0, -1.));
    lineVerts.push_back(glm::vec3(-1., 2.0,  1.));
    lineVerts.push_back(glm::vec3( 1., 2.0, -1.));
    lineVerts.push_back(glm::vec3( 1., 2.0,  1.));
    std::vector<std::pair<int, int>> lineInd;
    lineInd.push_back({ 0, 4 });
    lineInd.push_back({ 1, 5 });
    lineInd.push_back({ 2, 6 });
    lineInd.push_back({ 3, 7 });
    lineInd.push_back({ 0, 1 });
    lineInd.push_back({ 0, 2 });
    lineInd.push_back({ 1, 3 });
    lineInd.push_back({ 2, 3 });
    lineInd.push_back({ 4, 5 });
    lineInd.push_back({ 4, 6 });
    lineInd.push_back({ 5, 7 });
    lineInd.push_back({ 6, 7 });
    LineSegment lines(lineVerts, lineInd);
    // Shader debugDepth("debug_depth");


    // ShadowMapping PSM;
    while (!glfwWindowShouldClose(window))
    {
        //std::cout << io.WantCaptureMouse << std::endl;
        glfwPollEvents();
        
        for (int i = 0; i < 4; ++i) {
            lights[i].setPos(IG_lightPoses[i]);
            lights[i].setColor(IG_lightColors[i]);
        }
            
        std::vector<glm::vec3> lightPoses;
        std::vector<glm::vec3> lightColors;
        std::vector<unsigned int> depthMapIDs;
        std::vector<bool> lightsOn = { IG_lightsOn[0], IG_lightsOn[1], IG_lightsOn[2], IG_lightsOn[3]};
        std::vector<glm::mat4> lightSpaceMatrices;
        for (int i = 0; i < lights.size(); ++i)
        {
            // if (lightsOn[i]) lights[i].generateShadowMap();
            if (shadowMapping) depthMapIDs.push_back(lights[i].getDepthMap());
            lightPoses.push_back(lights[i].getPos());
            lightColors.push_back(lights[i].getColor());
            lightSpaceMatrices.push_back(lights[i].getlightSpaceMatrix());
        }

        camera.processInput(window);
        /************************* depth map *************************/
        for (int i = 0; i < lights.size(); ++i)
        {
            if (lightsOn[i] == false || !shadowMapping) continue;
            lights[i].generateShadowMap(meshes);
        }
        /************************* depth map *************************/
        // debugDepth.use();
        // debugDepth.setInt("depthMap", 0);
        // int width, height;
        // glfwGetWindowSize(window, &width, &height);
        // glViewport(0, 0, width, height);
        // glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        // debugDepth.setFloat("near_plane",0.1);
        // debugDepth.setFloat("far_plane",1000.);
        // glActiveTexture(GL_TEXTURE0);
        // glBindTexture(GL_TEXTURE_2D, lights[3].getDepthMap());
        // bunny1.getMesh()->Draw();
        // bunny2.getMesh()->Draw();
        // bunny3.getMesh()->Draw();

        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClearColor(clearColor.x, clearColor.y, clearColor.z, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 model, view, projection;
        camera.computeMVP(model, view, projection);
        {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
            floor.getMesh()->Draw(
                depthMapIDs,
                model, view, projection,
                lightPoses, lightColors,
                camera.getPos(),
                lightsOn,
                lightSpaceMatrices,
                shadowMapping
            );
            glDisable(GL_CULL_FACE);
            // floor.getMesh()->DrawEdge(model, view, projection, glm::vec3(0.));
        }
        {
            // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            bunny1.getMesh()->Draw(
                depthMapIDs,
                model, view, projection,
                lightPoses, lightColors,
                camera.getPos(),
                lightsOn,
                lightSpaceMatrices,
                shadowMapping
            );
            bunny1.getMesh()->DrawEdge(model, view, projection, glm::vec3(0.));
            bunny2.getMesh()->Draw(
                depthMapIDs,
                model, view, projection,
                lightPoses, lightColors,
                camera.getPos(),
                lightsOn,
                lightSpaceMatrices,
                shadowMapping
            );
            bunny3.getMesh()->Draw(
                depthMapIDs,
                model, view, projection,
                lightPoses, lightColors,
                camera.getPos(),
                lightsOn,
                lightSpaceMatrices,
                shadowMapping
            );
            /*Draw points*/
            // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            {
                GLint viewport[4];
                glGetIntegerv(GL_VIEWPORT, viewport);
                glm::vec4 __viewport;
                __viewport.x = 1.0f * viewport[0];
                __viewport.y = 1.0f * viewport[1];
                __viewport.z = 1.0f * viewport[2];
                __viewport.w = 1.0f * viewport[3];
                spheres.Draw(model, view, projection, lightPoses, lightColors, lightsOn, __viewport, camera.getPos(), 0.1f);
            }
            {
                lines.Draw(model, view, projection, glm::vec3(1.), lightsOn[0] || lightsOn[1] || lightsOn[2] || lightsOn[3]);
            }
        }
        {
            for (int i = 0; i < lights.size(); ++i)
            {
                if (lightsOn[i] == false || IG_hideLights[i] == true)
                    continue;
                auto light = lights[i];
                light.getMesh()->Draw(
                    model, view, projection,
                    light.getPos(), light.getColor(),
                    0.2f
                );
            }
        }
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Begin("Control Panel");
        // ImGui::Checkbox("background animation", &IG_bgAnimation);
        // ImGui::Checkbox("gradient background", &IG_gradientBG);
        ImGui::Checkbox("enable shadow mapping", &(shadowMapping));
        for (int i = 0; i < lights.size(); ++i)
        {
            if (ImGui::CollapsingHeader(("light#" + std::to_string(i)).c_str()))
            {
                ImGui::Checkbox(("light#" + std::to_string(i) + " on / off").c_str(), &(IG_lightsOn[i]));
                ImGui::Checkbox(("hide light#" + std::to_string(i)).c_str(), &(IG_hideLights[i]));
                ImGui::ColorEdit3(("light#" + std::to_string(i) + " color").c_str(), IG_lightColors[i]);
                ImGui::DragFloat(("light#" + std::to_string(i) + " position.x").c_str(), &IG_lightPoses[i][0], 0.05f);
                ImGui::DragFloat(("light#" + std::to_string(i) + " position.y").c_str(), &IG_lightPoses[i][1], 0.05f);
                ImGui::DragFloat(("light#" + std::to_string(i) + " position.z").c_str(), &IG_lightPoses[i][2], 0.05f);
            }
        }
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        // glfwPollEvents();
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

#include "scene.hpp"

bool scene::isShadowMappingCubic = false;

scene::scene(std::string _name, bool _isShadowMappingCubic)
{
    name = _name;
    isShadowMappingCubic = _isShadowMappingCubic;
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);
#endif

    window = glfwCreateWindow(width, height, name.c_str(), nullptr, nullptr);
    camera = std::make_shared<QuatCamera>(window);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, QuatCamera::framebuffer_size_callback);
    glfwSetScrollCallback(window, QuatCamera::scroll_callback);
    glfwSetKeyCallback(window, QuatCamera::keyboard_callback);
    glfwSetMouseButtonCallback(window, QuatCamera::mousebutton_callback);
    glfwSetCursorPosCallback(window, QuatCamera::cursor_callback);

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}


void scene::lightingSetUp()
{
    if (fakeFloor)
    {
        floor = std::make_shared<Floor>(100.0f, isShadowMappingCubic);
        meshes.push_back(floor->getMesh());
    }
    for(unsigned int i = 0; i < static_mesh_loaders.size(); ++i)
        meshes.push_back(static_mesh_loaders[i].getMesh());
    lights.push_back(Light(glm::vec3(-3.0f, 3.0f, -3.0f), glm::vec3(1.0, 0.0, 0.0), meshes, isShadowMappingCubic));
    lights.push_back(Light(glm::vec3(-3.0f, 3.0f, 3.0f), glm::vec3(0.0, 1.0, 0.0), meshes, isShadowMappingCubic));
    lights.push_back(Light(glm::vec3(3.0f, 3.0f, -3.0f), glm::vec3(0.0, 0.0, 1.0), meshes, isShadowMappingCubic));
    lights.push_back(Light(glm::vec3(3.0f, 3.0f, 3.0f), glm::vec3(1.0, 1.0, 1.0), meshes, isShadowMappingCubic));
}


void scene::updateLighting()
{
    lightPoses.clear(); lightColors.clear(); depthMapIDs.clear(); lightSpaceMatrices.clear();
    lightsOn = { IG_lightsOn[0], IG_lightsOn[1], IG_lightsOn[2], IG_lightsOn[3] };
    for (int i = 0; i < nLights; ++i)
    {
        lights[i].shadowMeshes = meshes;
        lights[i].setPos(IG_lightPoses[i]);
        lights[i].setColor(IG_lightColors[i]);
        if (shadowMapping) depthMapIDs.push_back(lights[i].getDepthMap());
        lightPoses.push_back(lights[i].getPos());
        lightColors.push_back(lights[i].getColor());
        lightSpaceMatrices.push_back(lights[i].getlightSpaceMatrix());
        if (lightsOn[i] == false || !shadowMapping) continue;
        lights[i].generateShadowMap();
    }
}


void scene::loadSurfaceMeshFromOBJ(std::string inputfile, float scale,
		glm::vec3 translate, glm::vec3 rotate, // pitch, yaw, roll
		std::vector<glm::vec3> material)
{
    static_mesh_loaders.emplace_back(inputfile, scale, translate, rotate, material, isShadowMappingCubic);
}


void scene::loadSurfaceMeshFromOBJ(std::string inputfile, float scale,
		glm::vec3 translate, glm::vec3 rotate, // pitch, yaw, roll
		std::string albedo, std::string metallic,
		std::string normal, std::string roughness, std::string ao)
{
    static_mesh_loaders.emplace_back(inputfile, scale, translate, rotate, isShadowMappingCubic, albedo, metallic, normal, roughness, ao);
}

void scene::loadStaticParticles(glm::vec3 lower_bound, glm::vec3 upper_bound, float delta,
    float radius, glm::vec3 color, glm::vec2 material)
{
    std::vector<Point> points;
    for (float x = lower_bound.x; x <= upper_bound.x; x += delta) for (float y = lower_bound.y; y <= upper_bound.y; y += delta) for (float z = lower_bound.z; z <= upper_bound.z; z += delta)
    {
        points.push_back({ glm::vec3(x, y, z), color });
    }
    std::shared_ptr<Sphere> spheres = std::make_shared<Sphere>(points, material);
    spheres->radius = radius;
    particles.push_back(spheres);
}

void scene::drawLights()
{
    //glEnable(GL_CULL_FACE);
    //glCullFace(GL_BACK);

    for (int i = 0; i < nLights; ++i)
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


void scene::drawMeshes()
{
    //glEnable(GL_CULL_FACE);
    //glCullFace(GL_BACK);
    if (isShadowMappingCubic)
    {
        for (unsigned int i = 0; i < meshes.size(); ++i)
        {
            //if (i == 0)
            //{
            //    glEnable(GL_CULL_FACE);
            //    glCullFace(GL_BACK);
            //}
            meshes[i]->Draw(
                depthMapIDs,
                model, view, projection,
                lightPoses, lightColors,
                camera->getPos(),
                lightsOn,
                lights[0].shadowMapping->far_plane,
                shadowMapping
            );
            //if (i == 0)
            //{
            //    glDisable(GL_CULL_FACE);
            //}
        }
    }
    else
    {
        for (unsigned int i = 0; i < meshes.size(); ++i)
        {
            //if (i == 0)
            //{
            //    glEnable(GL_CULL_FACE);
            //    glCullFace(GL_BACK);
            //}
            meshes[i]->Draw(
                depthMapIDs,
                model, view, projection,
                lightPoses, lightColors,
                camera->getPos(),
                lightsOn,
                lightSpaceMatrices,
                shadowMapping
            );
            //if (i == 0)
            //{
            //    glDisable(GL_CULL_FACE);
            //}
        }
    }
    if (showEdge)
    {
        unsigned int start_i = fakeFloor ? 1 : 0;
        for (unsigned int i = start_i; i < meshes.size(); ++i)
        {
            meshes[i]->DrawEdge(model, view, projection, glm::vec3(0.));
        }
    }
    //glDisable(GL_CULL_FACE);
}

void scene::drawLines()
{
    for (unsigned int i = 0; i < static_lines.size(); ++i)
    {
        static_lines[i]->Draw(model, view, projection, line_color, lightsOn[0] || lightsOn[1] || lightsOn[2] || lightsOn[3]);
    }
}

void scene::setLines(LineSegment lines, glm::vec3 color)
{
    static_lines.push_back(std::make_shared<LineSegment>(lines));
    line_color = color;
}

void scene::drawParticles()
{
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    glm::vec4 __viewport;
    __viewport.x = viewport[0];
    __viewport.y = viewport[1];
    __viewport.z = viewport[2];
    __viewport.w = viewport[3];
    for (unsigned int i = 0; i < particles.size(); ++i)
    {
        particles[i]->Draw(
            model, view, projection,
            lightPoses, lightColors, lightsOn,
            __viewport, camera->getPos()
        );
    }
}

scene::~scene()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

void scene::setImGUI(float fps)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::Begin("Control Panel");
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
    if (fps == 0)
    {
        ImGui::Text("frame rate: %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    }
    else
    {
        ImGui::Text("simulation frame rate: %f FPS)", fps);
    }
    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void scene::run()
{
    lightingSetUp();
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        camera->processInput(window);

        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClearColor(clearColor.x, clearColor.y, clearColor.z, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        camera->computeMVP(model, view, projection);
        drawLights();
        drawMeshes();
        drawLines();
        drawParticles();
        updateLighting();
        scene::setImGUI();

        glfwSwapBuffers(window);
    }
}

#pragma once
#ifndef PI
#define PI 3.1415926535897932
#endif

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <stdio.h>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <iostream>
#include <GLFW/glfw3.h>

#include "imgui.h"



class OrbitControl
{
public:
    OrbitControl(GLFWwindow* window, float YAW = - PI * 0.1, float PITCH = - PI * 0.1, float dist2Target = 10.0);

    static void framebuffer_size_callback(GLFWwindow* window, int widthIn, int heightIn);
    static void mousebutton_callback(GLFWwindow* window, int button, int action, int mods);
    static void cursor_callback(GLFWwindow* window, double xposIn, double yposIn);
    static void scroll_callback(GLFWwindow* window, double xoffsetIn, double yoffsetIn);
    static void keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void NDC(float* xpos, float* ypos);
    static glm::vec3 rotateVecQuat(glm::quat q, glm::vec3 v);
    static int getWidth();
    static int getHeight();

    void processInput(GLFWwindow* window);
    void computeMVP(glm::mat4& model, glm::mat4& view, glm::mat4& projection);
    glm::vec3 getPos();
    glm::vec3 getFront();
    glm::mat4 getView();
private:
    static GLFWwindow* win;
    static int width, height;
    static glm::vec3 cameraPos;
    static glm::vec3 cameraFront;
    static glm::vec3 cameraTarget;
    //static glm::vec3 cameraDirection;
    //static glm::vec3 cameraRight;
    static glm::vec3 cameraUp;
    static float yaw;
    static float pitch;
    static float lastX;
    static float lastY;
    static bool firstMouse;
    static float zoom;
    static double deltaTime;
    static double lastFrame;
    static bool buttons[];
    static float xoffset, yoffset;
    static const float scroll_sensitivity;
    static const float cursor_sensitivity;
    static const float WASD_sensitivity;
    static float mouse_left_sensitivity;
    static float mouse_right_sensitivity;
    float initPITCH, initYAW, initD2T;
public:
    static unsigned int space_count;
    static unsigned int sim_mode;
};

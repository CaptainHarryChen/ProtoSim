#pragma once
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <Camera/Camera.h>
#include <Object/Object.h>

class OrbitCamera : public Camera, public Object
{
public:
    OrbitCamera(float YAW = - glm::pi<float>() * 0.1, float PITCH = - glm::pi<float>() * 0.1, float dist2Target = 10.0);

    virtual CameraInfo GetCameraInfo() override;
    virtual void ProcessEvent(const Event& event) override;
    virtual void Update(double delta_time) override;

    void NDC(float* xpos, float* ypos);
    glm::vec3 rotateVecQuat(glm::quat q, glm::vec3 v);
    int getWidth();
    int getHeight();

    void computeMVP(glm::mat4& view, glm::mat4& projection);
    glm::vec3 getPos();
    glm::vec3 getFront();
    glm::mat4 getView();

    bool buttons[GLFW_MOUSE_BUTTON_LAST] = { 0 };
    glm::vec3 cameraPos;
    glm::vec3 cameraFront;
    glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 cameraUp = glm::vec3(0.0, 1.0, 0.0);
    float yaw;
    float pitch;
    float lastX;
    float lastY;
    float zoom = 45.0f;
    double deltaTime = 0.0f;
    double lastFrame = 0.0f;
    bool  firstMouse = true;
    float xoffset = 0.0f;
    float yoffset = 0.0f;
    const float scroll_sensitivity = 1.0f;
    const float cursor_sensitivity = 10.0f;
    const float WASD_sensitivity = 2.5f;
    float mouse_left_sensitivity = 1.0f;
    float mouse_right_sensitivity = 1.0f;
    float initPITCH = 0.0f;
    float initYAW= 0.0f;
    float initD2T = 10.0f;

protected:
    bool m_move_up = false;
    bool m_move_down = false;
    bool m_move_left = false;
    bool m_move_right = false;
    bool m_move_forward = false;
    bool m_move_backward = false;
};

#include "OrbitCamera.h"
#include <glm/gtc/quaternion.hpp>

OrbitCamera::OrbitCamera(float YAW, float PITCH, float dist2Target)
{
    initYAW = YAW; initPITCH = PITCH; initD2T = dist2Target;

    yaw = YAW;
    pitch = PITCH;

    glm::quat q(glm::vec3(pitch, yaw, 0));
    cameraUp = rotateVecQuat(q, glm::vec3(0.0, 1.0, 0.0));
    cameraFront = rotateVecQuat(q, glm::vec3(0.0, 0.0, -1.0));
    cameraPos = cameraTarget - dist2Target * cameraFront;
}

CameraInfo OrbitCamera::GetCameraInfo()
{
    CameraInfo camera_info;
    computeMVP(camera_info.view, camera_info.projection);
    camera_info.view_pos = cameraPos;
    camera_info.viewport = {0.0f, 0.0f, static_cast<float>(viewport_width), static_cast<float>(viewport_height)};
    return camera_info;
}

void OrbitCamera::ProcessEvent(const Event& event)
{
    switch(event.type)
    {
    case EventType::FramebufferSize:
    {
        this->viewport_height = event.framebuffer_size.height;
        this->viewport_width = event.framebuffer_size.width;
        break;
    }
    case EventType::MouseButton:
    {
        if (event.mouse_button.action != GLFW_RELEASE)
        {
            if (!buttons[event.mouse_button.button]) 
                buttons[event.mouse_button.button] = true;
            else 
                buttons[event.mouse_button.button] = false;
        }
        else
        {
            buttons[event.mouse_button.button] = false;
        }
        break;
    }
    case EventType::Cursor:
    {
        float xpos = static_cast<float>(event.cursor.xpos);
        float ypos = static_cast<float>(event.cursor.ypos);
        NDC(&xpos, &ypos);
        if (firstMouse)
        {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }
        xoffset = xpos - lastX;
        yoffset = lastY - ypos; 
        lastX = xpos;
        lastY = ypos;
        xoffset *= cursor_sensitivity;
        yoffset *= cursor_sensitivity;
        if (buttons[GLFW_MOUSE_BUTTON_LEFT])
        {
            yaw += xoffset * mouse_left_sensitivity;
            pitch += yoffset * mouse_left_sensitivity;

            glm::quat q(glm::vec3(pitch, yaw, 0));
            cameraUp = rotateVecQuat(q, glm::vec3(0.0, 1.0, 0.0));
            cameraFront = rotateVecQuat(q, glm::vec3(0.0, 0.0, -1.0));
            cameraPos = cameraTarget - glm::distance(cameraPos, cameraTarget) * cameraFront;
        }
        if (buttons[GLFW_MOUSE_BUTTON_RIGHT])
        {
            glm::vec3 cameraRight = glm::normalize(glm::cross(cameraFront, cameraUp));
            cameraPos += xoffset * cameraRight * mouse_right_sensitivity;
            cameraPos -= yoffset * cameraUp * mouse_right_sensitivity;
            cameraTarget += xoffset * cameraRight * mouse_right_sensitivity;
            cameraTarget -= yoffset * cameraUp * mouse_right_sensitivity;
        }
        break;
    }
    case EventType::Scroll:
    {
        cameraPos += glm::vec3((float)event.scroll.yoffset) * cameraFront * scroll_sensitivity;
        cameraTarget += glm::vec3((float)event.scroll.yoffset) * cameraFront * scroll_sensitivity;
        break;
    }
    case EventType::Key:
    {
        switch(event.key.key)
        {
        case GLFW_KEY_W:
            m_move_up = event.key.action != GLFW_RELEASE;
            break;
        case GLFW_KEY_S:
            m_move_down = event.key.action != GLFW_RELEASE;
            break;
        case GLFW_KEY_A:
            m_move_left = event.key.action != GLFW_RELEASE;
            break;
        case GLFW_KEY_D:
            m_move_right = event.key.action != GLFW_RELEASE;
            break;
        case GLFW_KEY_LEFT_SHIFT:
            m_move_forward = event.key.action != GLFW_RELEASE;
            break;
        case GLFW_KEY_LEFT_CONTROL:
            m_move_backward = event.key.action != GLFW_RELEASE;
            break;
        case GLFW_KEY_TAB:
            if (event.key.action == GLFW_RELEASE)
            {
                firstMouse = true;
                yaw = initYAW;
                pitch = initPITCH;
                glm::quat q(glm::vec3(pitch, yaw, 0));
                cameraUp = rotateVecQuat(q, glm::vec3(0.0, 1.0, 0.0));
                cameraFront = rotateVecQuat(q, glm::vec3(0.0, 0.0, -1.0));
                cameraTarget = glm::vec3(0.0f);
                cameraPos = cameraTarget - initD2T * cameraFront;
            }
            break;
        }
    }
    }
}

void OrbitCamera::Update(double delta_time)
{
    glm::vec3 cameraRight = glm::normalize(glm::cross(cameraFront, cameraUp));
    float cameraSpeed = WASD_sensitivity * (float)delta_time;
    if (m_move_up)
        cameraPos += cameraSpeed * cameraUp, cameraTarget += cameraSpeed * cameraUp;
    if (m_move_down)
        cameraPos -= cameraSpeed * cameraUp, cameraTarget -= cameraSpeed * cameraUp;
    if (m_move_left)
        cameraPos -= cameraSpeed * cameraRight, cameraTarget -= cameraSpeed * cameraRight;
    if (m_move_right)
        cameraPos += cameraSpeed * cameraRight, cameraTarget += cameraSpeed * cameraRight;
    if (m_move_forward)
        cameraPos += cameraSpeed * cameraFront, cameraTarget += cameraSpeed * cameraFront;
    if (m_move_backward)
        cameraPos -= cameraSpeed * cameraFront, cameraTarget -= cameraSpeed * cameraFront;
}

void OrbitCamera::NDC(float* xpos, float* ypos)
{
    *xpos = float(*xpos) / float(viewport_width);
    *ypos = float(viewport_height - *ypos) / float(viewport_height);
}

glm::vec3 OrbitCamera::rotateVecQuat(glm::quat q, glm::vec3 v)
{
    glm::vec3 u(q.x, q.y, q.z);
    float s = q.w;
    return 2.0f * glm::dot(u, v) * u + (s * s - glm::dot(u, u)) * v + 2.0f * s * glm::cross(u, v);
}

void OrbitCamera::computeMVP(glm::mat4& view, glm::mat4& projection)
{
    view = glm::lookAt(cameraPos, cameraTarget, cameraUp);

    projection = glm::perspective(glm::radians(zoom), static_cast<float>(viewport_width) / static_cast<float>(viewport_height), 0.1f, 100.0f);
}

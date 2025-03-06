#include "QuatCamera.h"

GLFWwindow* QuatCamera::win = nullptr;
int QuatCamera::width;
int QuatCamera::height;
bool QuatCamera::buttons[GLFW_MOUSE_BUTTON_LAST] = { 0 };
glm::vec3 QuatCamera::cameraPos;
glm::vec3 QuatCamera::cameraFront;
glm::vec3 QuatCamera::cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 QuatCamera::cameraUp = glm::vec3(0.0, 1.0, 0.0);
float QuatCamera::yaw;
float QuatCamera::pitch;
float QuatCamera::lastX;
float QuatCamera::lastY;
float QuatCamera::zoom = 45.0f;
double QuatCamera::deltaTime = 0.0f;
double QuatCamera::lastFrame = 0.0f;
bool  QuatCamera::firstMouse = true;
float QuatCamera::xoffset = 0.0f;
float QuatCamera::yoffset = 0.0f;
const float QuatCamera::scroll_sensitivity = 1.0f;
const float QuatCamera::cursor_sensitivity = 10.0f;
const float QuatCamera::WASD_sensitivity = 2.5f;
float QuatCamera::mouse_left_sensitivity = 1.0f;
float QuatCamera::mouse_right_sensitivity = 1.0f;
unsigned int QuatCamera::space_count = 0;
unsigned int QuatCamera::sim_mode = 0;

QuatCamera::QuatCamera(GLFWwindow* window, float YAW, float PITCH, float dist2Target)
{
    initYAW = YAW; initPITCH = PITCH; initD2T = dist2Target;

    win = window;
    glfwGetWindowSize(win, &width, &height);

    yaw = YAW;
    pitch = PITCH;

    glm::quat q(glm::vec3(pitch, yaw, 0));
    cameraUp = rotateVecQuat(q, glm::vec3(0.0, 1.0, 0.0));
    cameraFront = rotateVecQuat(q, glm::vec3(0.0, 0.0, -1.0));
    cameraPos = cameraTarget - dist2Target * cameraFront;
}

void QuatCamera::framebuffer_size_callback(GLFWwindow* window, int widthIn, int heightIn)
{
    glViewport(0, 0, widthIn, heightIn);
    width = widthIn;
    height = heightIn;
}

void QuatCamera::mousebutton_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (action != GLFW_RELEASE)
        if (!buttons[button]) buttons[button] = true;
        else buttons[button] = false;
    else buttons[button] = false;
}

void QuatCamera::cursor_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    if (ImGui::GetIO().WantCaptureMouse) return;
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);
    NDC(&xpos, &ypos);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    xoffset = xpos - lastX;
    yoffset = lastY - ypos; // reversed: y ranges bottom to top
    lastX = xpos;
    lastY = ypos;

    xoffset *= cursor_sensitivity;
    yoffset *= cursor_sensitivity;
    if (buttons[GLFW_MOUSE_BUTTON_LEFT])
    {
        //std::cout << yaw << " " << pitch << std::endl;

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
}

void QuatCamera::scroll_callback(GLFWwindow* window, double xoffsetIn, double yoffsetIn)
{
    if (ImGui::GetIO().WantCaptureMouse) return;
    //zoom -= scroll_sensitivity * static_cast<float>(yoffsetIn);
    cameraPos += glm::vec3((float)yoffsetIn) * cameraFront * scroll_sensitivity;
    cameraTarget += glm::vec3((float)yoffsetIn) * cameraFront * scroll_sensitivity;
}

void QuatCamera::keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{

    if (action != GLFW_PRESS)
        return;

    switch (key)
    {
    case GLFW_KEY_SPACE:
        space_count += 1;
        break;
    case GLFW_KEY_M:
        sim_mode += 1;
        break;
    default:
        break;
    }
}

void QuatCamera::NDC(float* xpos, float* ypos)
{
    *xpos = float(*xpos) / float(width);
    *ypos = float(height - *ypos) / float(height);
    //*xpos = float(*xpos) / float(width) - 0.5f;
    //*ypos = float(height - *ypos) / float(height) - 0.5f;
}

glm::vec3 QuatCamera::rotateVecQuat(glm::quat q, glm::vec3 v)
{
    glm::vec3 u(q.x, q.y, q.z);
    float s = q.w;
    return 2.0f * glm::dot(u, v) * u + (s * s - glm::dot(u, u)) * v + 2.0f * s * glm::cross(u, v);
}

int QuatCamera::getWidth() { return width; }
int QuatCamera::getHeight() { return height; }


void QuatCamera::processInput(GLFWwindow* window)
{
    double currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    glm::vec3 cameraRight = glm::normalize(glm::cross(cameraFront, cameraUp));
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    float cameraSpeed = WASD_sensitivity * (float)deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraUp, cameraTarget += cameraSpeed * cameraUp;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraUp, cameraTarget -= cameraSpeed * cameraUp;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraRight, cameraTarget -= cameraSpeed * cameraRight;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraRight, cameraTarget += cameraSpeed * cameraRight;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront, cameraTarget += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront, cameraTarget -= cameraSpeed * cameraFront;
        

    //if (buttons[GLFW_MOUSE_BUTTON_LEFT])
    //{
    //    //std::cout << yaw << " " << pitch << std::endl;

    //    yaw += xoffset * mouse_left_sensitivity;
    //    pitch += yoffset * mouse_left_sensitivity;

    //    glm::quat q(glm::vec3(pitch, yaw, 0));
    //    cameraUp = rotateVecQuat(q, glm::vec3(0.0, 1.0, 0.0));
    //    cameraFront = rotateVecQuat(q, glm::vec3(0.0, 0.0, -1.0));
    //    cameraPos = cameraTarget - glm::distance(cameraPos, cameraTarget) * cameraFront;
    //}

    //if (buttons[GLFW_MOUSE_BUTTON_RIGHT])
    //{
    //    cameraPos += xoffset * cameraRight * mouse_right_sensitivity;
    //    cameraPos -= yoffset * cameraUp * mouse_right_sensitivity;
    //    cameraTarget += xoffset * cameraRight * mouse_right_sensitivity;
    //    cameraTarget -= yoffset * cameraUp * mouse_right_sensitivity;
    //}

    if (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS)
    {
        //std::cout << "Reset" << std::endl;
        yaw = initYAW;
        pitch = initPITCH;

        glm::quat q(glm::vec3(pitch, yaw, 0));
        cameraUp = rotateVecQuat(q, glm::vec3(0.0, 1.0, 0.0));
        cameraFront = rotateVecQuat(q, glm::vec3(0.0, 0.0, -1.0));
        cameraTarget = glm::vec3(0.0f);
        cameraPos = cameraTarget - initD2T * cameraFront;
    }

    //xoffset = 0.;
    //yoffset = 0.;
}

void QuatCamera::computeMVP(glm::mat4& model, glm::mat4& view, glm::mat4& projection)
{
    model = glm::mat4(1.0f);
    //model = glm::rotate(model, glm::radians(15.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    view = getView();

    projection = glm::perspective(glm::radians(zoom), static_cast<float>(width) / static_cast<float>(height), 0.1f, 100.0f);
}

glm::vec3 QuatCamera::getPos()
{
    return cameraPos;
}

glm::vec3 QuatCamera::getFront()
{
    return cameraFront;
}

glm::mat4 QuatCamera::getView()
{
    return glm::lookAt(cameraPos, cameraTarget, cameraUp);
}

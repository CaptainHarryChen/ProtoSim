#include "OrbitControl.h"

GLFWwindow* OrbitControl::win = nullptr;
int OrbitControl::width;
int OrbitControl::height;
bool OrbitControl::buttons[GLFW_MOUSE_BUTTON_LAST] = { 0 };
glm::vec3 OrbitControl::cameraPos;
glm::vec3 OrbitControl::cameraFront;
glm::vec3 OrbitControl::cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 OrbitControl::cameraUp = glm::vec3(0.0, 1.0, 0.0);
float OrbitControl::yaw;
float OrbitControl::pitch;
float OrbitControl::lastX;
float OrbitControl::lastY;
float OrbitControl::zoom = 45.0f;
double OrbitControl::deltaTime = 0.0f;
double OrbitControl::lastFrame = 0.0f;
bool  OrbitControl::firstMouse = true;
float OrbitControl::xoffset = 0.0f;
float OrbitControl::yoffset = 0.0f;
const float OrbitControl::scroll_sensitivity = 1.0f;
const float OrbitControl::cursor_sensitivity = 10.0f;
const float OrbitControl::WASD_sensitivity = 2.5f;
float OrbitControl::mouse_left_sensitivity = 1.0f;
float OrbitControl::mouse_right_sensitivity = 1.0f;
unsigned int OrbitControl::space_count = 0;
unsigned int OrbitControl::sim_mode = 0;

OrbitControl::OrbitControl(GLFWwindow* window, float YAW, float PITCH, float dist2Target)
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

void OrbitControl::framebuffer_size_callback(GLFWwindow* window, int widthIn, int heightIn)
{
    // glViewport(0, 0, widthIn, heightIn);
    width = widthIn;
    height = heightIn;
}

void OrbitControl::mousebutton_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (action != GLFW_RELEASE)
        if (!buttons[button]) buttons[button] = true;
        else buttons[button] = false;
    else buttons[button] = false;
}

void OrbitControl::cursor_callback(GLFWwindow* window, double xposIn, double yposIn)
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

void OrbitControl::scroll_callback(GLFWwindow* window, double xoffsetIn, double yoffsetIn)
{
    if (ImGui::GetIO().WantCaptureMouse) return;
    //zoom -= scroll_sensitivity * static_cast<float>(yoffsetIn);
    cameraPos += glm::vec3((float)yoffsetIn) * cameraFront * scroll_sensitivity;
    cameraTarget += glm::vec3((float)yoffsetIn) * cameraFront * scroll_sensitivity;
}

void OrbitControl::keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
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

void OrbitControl::NDC(float* xpos, float* ypos)
{
    *xpos = float(*xpos) / float(width);
    *ypos = float(height - *ypos) / float(height);
    //*xpos = float(*xpos) / float(width) - 0.5f;
    //*ypos = float(height - *ypos) / float(height) - 0.5f;
}

glm::vec3 OrbitControl::rotateVecQuat(glm::quat q, glm::vec3 v)
{
    glm::vec3 u(q.x, q.y, q.z);
    float s = q.w;
    return 2.0f * glm::dot(u, v) * u + (s * s - glm::dot(u, u)) * v + 2.0f * s * glm::cross(u, v);
}

int OrbitControl::getWidth() { return width; }
int OrbitControl::getHeight() { return height; }


void OrbitControl::processInput(GLFWwindow* window)
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

void OrbitControl::computeMVP(glm::mat4& model, glm::mat4& view, glm::mat4& projection)
{
    model = glm::mat4(1.0f);
    //model = glm::rotate(model, glm::radians(15.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    view = getView();

    projection = glm::perspective(glm::radians(zoom), static_cast<float>(width) / static_cast<float>(height), 0.1f, 100.0f);
}

glm::vec3 OrbitControl::getPos()
{
    return cameraPos;
}

glm::vec3 OrbitControl::getFront()
{
    return cameraFront;
}

glm::mat4 OrbitControl::getView()
{
    return glm::lookAt(cameraPos, cameraTarget, cameraUp);
}

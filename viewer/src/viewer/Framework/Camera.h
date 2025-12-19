#pragma once
#include <glm/glm.hpp>

namespace viewer {

struct CameraInfo
{
    glm::mat4 view;
    glm::mat4 projection;
    glm::vec3 view_pos;
    glm::vec4 viewport;
};

class Camera
{
public:
    Camera()          = default;
    virtual ~Camera() = default;

    virtual CameraInfo GetCameraInfo() = 0;

    int viewport_width;
    int viewport_height;
};

}  // namespace viewer

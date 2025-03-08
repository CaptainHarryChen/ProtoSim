#pragma once
#include <glm/glm.hpp>

struct LightInfo
{
    glm::vec3 pos;
    glm::vec3 color;
    bool isOn;
};

class Light
{
public:
    Light() = default;
    virtual ~Light() = default;

    virtual LightInfo GetLightInfo() = 0;
};

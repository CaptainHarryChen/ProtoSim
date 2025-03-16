#pragma once
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <Object/Object.h>

class LineSegment;

class CubeLineBox : public Object
{
public:
    CubeLineBox(const std::vector<float> &bbox, float dist, glm::vec3 color);

protected:
    std::shared_ptr<LineSegment> m_line_segment;
};

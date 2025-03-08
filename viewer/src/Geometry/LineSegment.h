#pragma once
#include <vector>
#include <memory>
#include <Render/RenderObject.h>

struct LineSeg
{
	glm::vec3 a, b;
};

class LineSegment : public RenderObject
{
public:
	LineSegment(std::vector<glm::vec3> vertices, std::vector<std::pair<int, int>> connectivity);
	virtual ~LineSegment() = default;

	std::vector<LineSeg> m_line_segs;

	virtual void DrawVAO() const override;

protected:
	unsigned int m_VAO;
	unsigned int m_VBO;
};

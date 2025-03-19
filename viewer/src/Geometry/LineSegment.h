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
	LineSegment(const std::vector<LineSeg> &line_segs);
	LineSegment(const std::vector<glm::vec3> &vertices, const std::vector<std::pair<int, int>> &connectivity);
	virtual ~LineSegment() = default;

	std::vector<LineSeg> m_line_segs;

	virtual void DrawVAO() const override;
	virtual inline unsigned int GetVBO() const { return m_VBO; }

protected:
	unsigned int m_VAO;
	unsigned int m_VBO;
};

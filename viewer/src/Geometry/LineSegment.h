#pragma once
#include <vector>
#include <memory>
#include <glm/glm.hpp>

class Shader;

struct LineSeg
{
	glm::vec3 a, b;
};

class LineSegment
{
public:
	LineSegment(std::vector<glm::vec3> vertices, std::vector<std::pair<int, int>> connectivity);

	void Draw(); // outer call for shadow-mapping
	void Draw(glm::mat4 model,
			  glm::mat4 view,
			  glm::mat4 projection,
			  glm::vec3 color,
			  bool lightOn);

private:
	std::vector<LineSeg> buffer;
	unsigned int VAO;
	unsigned int VBO;
	std::shared_ptr<Shader> shader, shaderSphereRayCast, shaderSphere, shaderPoint;

	void setupGeometry();
};

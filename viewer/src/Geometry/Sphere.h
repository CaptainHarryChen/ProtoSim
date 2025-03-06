#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <memory>

class Shader;

struct Point
{
	glm::vec3 Position;
	glm::vec3 Color;
};

class Sphere
{
public:
	std::vector<Point> vertices;
	glm::vec2 material;
	unsigned int VAO;
	unsigned int VBO;
	float radius;

	Sphere(std::vector<Point> vertices, glm::vec2 material);

	void Draw(); // outer call for shadow-mapping

	void UpdateVertices(std::vector<Point> data);

	void Draw(
		glm::mat4 model,
		glm::mat4 view,
		glm::mat4 projection,
		std::vector<glm::vec3> lightPos,
		std::vector<glm::vec3> lightColor,
		std::vector<bool> lightsOn,
		glm::vec4 viewport,
		glm::vec3 viewPos);

	void Draw(
		glm::mat4 model,
		glm::mat4 view,
		glm::mat4 projection,
		std::vector<glm::vec3> lightPos,
		std::vector<glm::vec3> lightColor,
		std::vector<bool> lightsOn,
		glm::vec4 viewport,
		glm::vec3 viewPos,
		float _radius);

private:
	std::shared_ptr<Shader> shaderSphere;
	void setupGeometry();
};

#pragma once
#include <glad/glad.h>
#include <vector>
#include <glm/glm.hpp>
#include <memory>
#include "Shader.h"

struct LineSeg
{
	glm::vec3 a, b;
};

class LineSegment
{
public:
	unsigned int VAO;
	LineSegment(std::vector<glm::vec3> vertices, std::vector<std::pair<int,int>> connectivity)
	{
		for (int i = 0; i < connectivity.size(); ++i)
		{
			buffer.push_back({ vertices[connectivity[i].first],vertices[connectivity[i].second] });
		}
		setupGeometry();
		shader = std::make_shared<Shader>("line");
	}
	void Draw() // outer call for shadow-mapping
	{
		glBindVertexArray(VAO);
		glDrawArrays(GL_LINES, 0, 2 * (GLsizei)buffer.size());
		glBindVertexArray(0);
	}
	void Draw(
		glm::mat4 model,
		glm::mat4 view,
		glm::mat4 projection,
		glm::vec3 color,
		bool lightOn
	)
	{
		shader->use();
		shader->setMat4("model", model);
		shader->setMat4("view", view);
		shader->setMat4("projection", projection);
		shader->setVec3("color", color);
		shader->setBool("lightOn", lightOn);
		Draw();
	}

private:
	std::vector<LineSeg> buffer;
	unsigned int VBO;
	std::shared_ptr<Shader> shader, shaderSphereRayCast, shaderSphere, shaderPoint;
	void setupGeometry()
	{
		glGenVertexArrays(1, &VAO); glBindVertexArray(VAO);
		glGenBuffers(1, &VBO); glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, buffer.size() * sizeof(LineSeg), &buffer[0], GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glEnableVertexAttribArray(0);
		glBindVertexArray(0);
		glLineWidth(2.0);
	}
};

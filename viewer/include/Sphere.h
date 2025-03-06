#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Shader.h"

#include <string>
#include <vector>
#include <assert.h>
#include <stb_image.h>
#include <memory>


struct Point
{
	glm::vec3 Position;
	glm::vec3 Color;
};

class Spheres
{
public:
	std::vector<Point> vertices;
	glm::vec2 material;
	unsigned int VAO;
	unsigned int VBO;
	float radius;

	Spheres(std::vector<Point> vertices, glm::vec2 material)
	{

		this->vertices = vertices;
		this->material = material;
		setupGeometry();
		shaderSphere = std::make_shared<Shader>("sphere_raycast", true);
	}

	void Draw() // outer call for shadow-mapping
	{
		glBindVertexArray(VAO);
		glDrawArrays(GL_POINTS, 0, (GLsizei)vertices.size());
		//std::cout << vertices.size() << std::endl;
		glBindVertexArray(0);
	}

	void UpdateVertices(std::vector<Point> data)
	{
		glBindVertexArray(VAO);
		vertices = data;
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(Point), &vertices[0]);
		glBindVertexArray(0);
	}

	void Draw(
		glm::mat4 model,
		glm::mat4 view,
		glm::mat4 projection,
		std::vector<glm::vec3> lightPos,
		std::vector<glm::vec3> lightColor,
		std::vector<bool> lightsOn,
		glm::vec4 viewport,
		glm::vec3 viewPos
	)
	{
		shaderSphere->use();
		shaderSphere->setMat4("model", model);
		shaderSphere->setMat4("view", view);

		shaderSphere->setMat4("u_projMatrix", projection);
		shaderSphere->setMat4("u_invProjMatrix", glm::inverse(projection));
		shaderSphere->setVec4("u_viewport", viewport);
		shaderSphere->setFloat("u_pointRadius", radius);

		for (int i = 0; i < lightPos.size(); ++i)
		{
			shaderSphere->setVec3("lightPos[" + std::to_string(i) + "]", lightPos[i]);
			shaderSphere->setVec3("lightColor[" + std::to_string(i) + "]", lightColor[i]);
			shaderSphere->setBool("lightsOn[" + std::to_string(i) + "]", lightsOn[i]);
		}
		shaderSphere->setFloat("metalicIn", material[0]);
		shaderSphere->setFloat("roughnessIn", material[1]);

		shaderSphere->setVec3("viewPos", viewPos);

		Draw();
	}

	void Draw(
		glm::mat4 model,
		glm::mat4 view,
		glm::mat4 projection,
		std::vector<glm::vec3> lightPos,
		std::vector<glm::vec3> lightColor,
		std::vector<bool> lightsOn,
		glm::vec4 viewport,
		glm::vec3 viewPos,
		float _radius
	)
	{
		shaderSphere->use();
		shaderSphere->setMat4("model", model);
		shaderSphere->setMat4("view", view);

		shaderSphere->setMat4("u_projMatrix", projection);
		shaderSphere->setMat4("u_invProjMatrix", glm::inverse(projection));
		shaderSphere->setVec4("u_viewport", viewport);
		shaderSphere->setFloat("u_pointRadius", _radius);

		for (int i = 0; i < lightPos.size(); ++i)
		{
			shaderSphere->setVec3("lightPos[" + std::to_string(i) + "]", lightPos[i]);
			shaderSphere->setVec3("lightColor[" + std::to_string(i) + "]", lightColor[i]);
			shaderSphere->setBool("lightsOn[" + std::to_string(i) + "]", lightsOn[i]);
		}
		shaderSphere->setFloat("metalicIn", material[0]);
		shaderSphere->setFloat("roughnessIn", material[1]);

		shaderSphere->setVec3("viewPos", viewPos);

		Draw();
	}

private:
	std::shared_ptr<Shader> shaderSphere;
	void setupGeometry()
	{
		glGenVertexArrays(1, &VAO); glBindVertexArray(VAO);
		glGenBuffers(1, &VBO); glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Point), &vertices[0], GL_STATIC_DRAW);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Point), (void*)0);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Point), (void*)offsetof(Point, Color));
		glEnableVertexAttribArray(1);
		glBindVertexArray(0);
		glEnable(GL_PROGRAM_POINT_SIZE);
	}
};

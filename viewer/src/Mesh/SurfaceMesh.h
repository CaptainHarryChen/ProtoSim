#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <cassert>
#include <string>
#include <vector>

struct Vertex
{
	glm::vec3 Position;
	glm::vec3 Normal;
	glm::vec2 TexCoords;
};

class Shader;

class SurfaceMesh
{
public:
	SurfaceMesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<glm::vec3> material, bool cubic = true);
	SurfaceMesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<std::string> textures, bool cubic = true);
	SurfaceMesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices); // light

	std::vector<Vertex> vertices;
	std::vector<unsigned int> indices;
	std::vector<std::string> textures; // pbr texture
	std::vector<glm::vec3> material;   // pbr <albedo> <metallic, roughness, ao>

	unsigned int VAO;
	unsigned int VBO, EBO;

	void BindTexture();
	void ActivateTexture();
	void DrawEdge(glm::mat4 model, glm::mat4 view, glm::mat4 projection, glm::vec3 Color);
	void Draw();
	void UpdateVertices(std::vector<Vertex> data);

	// pbr, pbr texture
	void Draw(
		std::vector<unsigned int> depthMap3D,
		glm::mat4 model,
		glm::mat4 view,
		glm::mat4 projection,
		std::vector<glm::vec3> lightPos,
		std::vector<glm::vec3> lightColor,
		glm::vec3 viewPos,
		std::vector<bool> lightsOn,
		float far_plane,
		bool enableShadow = true);

	void Draw(
		std::vector<unsigned int> depthMap2D,
		glm::mat4 model,
		glm::mat4 view,
		glm::mat4 projection,
		std::vector<glm::vec3> lightPos,
		std::vector<glm::vec3> lightColor,
		glm::vec3 viewPos,
		std::vector<bool> lightsOn,
		std::vector<glm::mat4> lightSpaceMatrix,
		bool enableShadow = true);

	// light
	void Draw(glm::mat4 model, glm::mat4 view, glm::mat4 projection, glm::vec3 lightPos, glm::vec3 lightColor, float scale = 0.2f);

private:
	std::shared_ptr<Shader> shader;
	std::shared_ptr<Shader> shaderEdge;
	unsigned int aoID;
	unsigned int albedoID;
	unsigned int normalID;
	unsigned int metallicID;
	unsigned int roughnessID;
	bool textureBinded = false;
	bool isCubic = false;

	void setupMesh();
	unsigned int loadTexture(char const *path);
};

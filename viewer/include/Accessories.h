#pragma once

#include "MeshBase.h"

// rendering accessories
class Light: public MeshBase // render light cube + render shadow map
{
public:
	Light(glm::vec3 position, glm::vec3 color, std::vector<std::shared_ptr<SurfaceMesh>> meshes, bool isCubic = true)
	{
		// std::cout << isCubic << std::endl;
		lightPos = position;
		lightColor = color;
		shadowMeshes = meshes;
		shadowMapping = std::make_shared<ShadowMapping>(isCubic);
		mesh = std::make_shared<SurfaceMesh>(__CUBE_VERTICES, __CUBE_INDICES);
	}
	glm::vec3 getPos()
	{
		return lightPos;
	}
	
	unsigned int getDepthMap()
	{
		return shadowMapping->getDepthMap();
	}

	void generateShadowMap()
	{
		// std::cout << lightPos.x << " " << lightPos.y <<  " " << lightPos.z << std::endl;
		// std::cout << shadowMeshes.size() << std::endl;
		shadowMapping->Draw(lightPos, shadowMeshes);
	}

	glm::mat4 getlightSpaceMatrix()
	{
		return shadowMapping->getlightSpaceMatrix();
	}

	template<typename T>
	void setPos(T p)
	{
		lightPos = { p[0],p[1],p[2] };
	}

	template<typename T>
	void setColor(T c)
	{
		lightColor = { c[0],c[1],c[2] };
	}

	glm::vec3 getColor()
	{
		return lightColor;
	}
	std::shared_ptr<ShadowMapping> shadowMapping;
	std::vector<std::shared_ptr<SurfaceMesh>> shadowMeshes;
private:
	glm::vec3 lightPos;
	glm::vec3 lightColor;
	// std::shared_ptr<ShadowMapping> shadowMapping;
	// std::vector<std::shared_ptr<SurfaceMesh>> shadowMeshes;
	std::vector<Vertex> __CUBE_VERTICES = {
		// positions           // normals              // texture coords
		{{-0.5f, 0.0f, -0.5f}, { 0.0f,  0.0f, -1.0f},  {0.0f, 0.0f}},
		{{ 0.5f, 1.0f, -0.5f}, { 0.0f,  0.0f, -1.0f},  {1.0f, 1.0f}},
		{{ 0.5f, 0.0f, -0.5f}, { 0.0f,  0.0f, -1.0f},  {1.0f, 0.0f}},

		{{ 0.5f, 1.0f, -0.5f}, { 0.0f,  0.0f, -1.0f},  {1.0f, 1.0f}},
		{{-0.5f, 0.0f, -0.5f}, { 0.0f,  0.0f, -1.0f},  {0.0f, 0.0f}},
		{{-0.5f, 1.0f, -0.5f}, { 0.0f,  0.0f, -1.0f},  {0.0f, 1.0f}},

		{{-0.5f, 0.0f,  0.5f}, { 0.0f,  0.0f,  1.0f},  {0.0f, 0.0f}},
		{{ 0.5f, 0.0f,  0.5f}, { 0.0f,  0.0f,  1.0f},  {1.0f, 0.0f}},
		{{ 0.5f, 1.0f,  0.5f}, { 0.0f,  0.0f,  1.0f},  {1.0f, 1.0f}},

		{{ 0.5f, 1.0f,  0.5f}, { 0.0f,  0.0f,  1.0f},  {1.0f, 1.0f}},
		{{-0.5f, 1.0f,  0.5f}, { 0.0f,  0.0f,  1.0f},  {0.0f, 1.0f}},
		{{-0.5f, 0.0f,  0.5f}, { 0.0f,  0.0f,  1.0f},  {0.0f, 0.0f}},

		{{-0.5f, 1.0f,  0.5f}, {-1.0f,  0.0f,  0.0f},  {1.0f, 0.0f}},
		{{-0.5f, 1.0f, -0.5f}, {-1.0f,  0.0f,  0.0f},  {1.0f, 1.0f}},
		{{-0.5f, 0.0f, -0.5f}, {-1.0f,  0.0f,  0.0f},  {0.0f, 1.0f}},

		{{-0.5f, 0.0f, -0.5f}, {-1.0f,  0.0f,  0.0f},  {0.0f, 1.0f}},
		{{-0.5f, 0.0f,  0.5f}, {-1.0f,  0.0f,  0.0f},  {0.0f, 0.0f}},
		{{-0.5f, 1.0f,  0.5f}, {-1.0f,  0.0f,  0.0f},  {1.0f, 0.0f}},

		{{ 0.5f, 1.0f,  0.5f}, { 1.0f,  0.0f,  0.0f},  {1.0f, 0.0f}},
		{{ 0.5f, 0.0f, -0.5f}, { 1.0f,  0.0f,  0.0f},  {0.0f, 1.0f}},
		{{ 0.5f, 1.0f, -0.5f}, { 1.0f,  0.0f,  0.0f},  {1.0f, 1.0f}},

		{{ 0.5f, 0.0f, -0.5f}, { 1.0f,  0.0f,  0.0f},  {0.0f, 1.0f}},
		{{ 0.5f, 1.0f,  0.5f}, { 1.0f,  0.0f,  0.0f},  {1.0f, 0.0f}},
		{{ 0.5f, 0.0f,  0.5f}, { 1.0f,  0.0f,  0.0f},  {0.0f, 0.0f}},

		{{-0.5f, 0.0f, -0.5f}, { 0.0f, -1.0f,  0.0f},  {0.0f, 1.0f}},
		{{ 0.5f, 0.0f, -0.5f}, { 0.0f, -1.0f,  0.0f},  {1.0f, 1.0f}},
		{{ 0.5f, 0.0f,  0.5f}, { 0.0f, -1.0f,  0.0f},  {1.0f, 0.0f}},

		{{ 0.5f, 0.0f,  0.5f}, { 0.0f, -1.0f,  0.0f},  {1.0f, 0.0f}},
		{{-0.5f, 0.0f,  0.5f}, { 0.0f, -1.0f,  0.0f},  {0.0f, 0.0f}},
		{{-0.5f, 0.0f, -0.5f}, { 0.0f, -1.0f,  0.0f},  {0.0f, 1.0f}},

		{{-0.5f, 1.0f, -0.5f}, { 0.0f,  1.0f,  0.0f},  {0.0f, 1.0f}},
		{{ 0.5f, 1.0f,  0.5f}, { 0.0f,  1.0f,  0.0f},  {1.0f, 0.0f}},
		{{ 0.5f, 1.0f, -0.5f}, { 0.0f,  1.0f,  0.0f},  {1.0f, 1.0f}},

		{{ 0.5f, 1.0f,  0.5f}, { 0.0f,  1.0f,  0.0f},  {1.0f, 0.0f}},
		{{-0.5f, 1.0f, -0.5f}, { 0.0f,  1.0f,  0.0f},  {0.0f, 1.0f}},
		{{-0.5f, 1.0f,  0.5f}, { 0.0f,  1.0f,  0.0f},  {0.0f, 0.0f}}
	};

	std::vector<unsigned int> __CUBE_INDICES = {
		0,  1,  2,
		3,  4,  5,
		6,  7,  8,
		9,  10, 11,
		12, 13, 14,
		15, 16, 17,
		18, 19, 20,
		21, 22, 23,
		24, 25, 26,
		27, 28, 29,
		30, 31, 32,
		33, 34, 35
	};

};

class Floor: public MeshBase
{
public:
	Floor(float scale = 1.0, bool isCubic = true)
	{
		// std::cout << isCubic << std::endl;
		std::vector<Vertex> __vertices = __PLANE_VERTICES;
		for (int i = 0; i < __vertices.size(); ++i)
		{
			__vertices[i].Position *= scale;
			__vertices[i].TexCoords *= scale;
		}
		std::string albedo = std::string(VIEWER_DIR) + "/data/floor/albedo.png";
		std::string metallic = std::string(VIEWER_DIR) + "/data/floor/metallic.png";
		std::string normal = std::string(VIEWER_DIR) + "/data/floor/normal.png";
		std::string roughness = std::string(VIEWER_DIR) + "/data/floor/roughness.png";
		std::string ao = std::string(VIEWER_DIR) + "/data/floor/ao.png";
		std::vector<std::string> textures = { albedo, normal, metallic, roughness, ao };
		mesh = std::make_shared<SurfaceMesh>(__vertices, __PLANE_INDICES, textures, isCubic);
		mesh->BindTexture();
	}
private:
	std::vector<Vertex> __PLANE_VERTICES =
	{
		{{-5.0f,  0.0f,  5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
		{{ 5.0f,  0.0f,  5.0f}, {0.0f, 1.0f, 0.0f}, {5.0f, 0.0f}},
		{{-5.0f,  0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 5.0f}},
		{{ 5.0f,  0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {5.0f, 5.0f}}
	};

	std::vector<unsigned int> __PLANE_INDICES = {
		0, 1, 2,
		1, 3, 2
	};
};

class Background: public MeshBase
{
public:
	Background() { SetUp(); }

	Background(glm::vec3 t, glm::vec3 m, glm::vec3 b) { Update(t, m, b); SetUp(); }
	
	void Update(glm::vec3 t, glm::vec3 m, glm::vec3 b)
	{
		__SCREEN_T = std::make_shared<glm::vec3>(t);
		__SCREEN_M = std::make_shared<glm::vec3>(m);
		__SCREEN_B = std::make_shared<glm::vec3>(b);

		float _[] =
		{
			-1.0f, -1.0f, __SCREEN_B->x, __SCREEN_B->y, __SCREEN_B->z,
			 1.0f, -1.0f, __SCREEN_B->x, __SCREEN_B->y, __SCREEN_B->z,
			-1.0f,  0.0f, __SCREEN_M->x, __SCREEN_M->y, __SCREEN_M->z,

			-1.0f,  0.0f, __SCREEN_M->x, __SCREEN_M->y, __SCREEN_M->z,
			 1.0f, -1.0f, __SCREEN_B->x, __SCREEN_B->y, __SCREEN_B->z,
			 1.0f,  0.0f, __SCREEN_M->x, __SCREEN_M->y, __SCREEN_M->z,

			-1.0f,  0.0f, __SCREEN_M->x, __SCREEN_M->y, __SCREEN_M->z,
			 1.0f,  0.0f, __SCREEN_M->x, __SCREEN_M->y, __SCREEN_M->z,
			-1.0f,  1.0f, __SCREEN_T->x, __SCREEN_T->y, __SCREEN_T->z,

			-1.0f,  1.0f, __SCREEN_T->x, __SCREEN_T->y, __SCREEN_T->z,
			 1.0f,  0.0f, __SCREEN_M->x, __SCREEN_M->y, __SCREEN_M->z,
			 1.0f,  1.0f, __SCREEN_T->x, __SCREEN_T->y, __SCREEN_T->z
		};
		for (int i = 0; i < 60; ++i)
			__SCREEN_VERTICES[i] = _[i];
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(__SCREEN_VERTICES), __SCREEN_VERTICES);
	}

	void Draw()
	{
		glDepthFunc(GL_LEQUAL);
		shader->use();
		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLES, 0, 12);
		glBindVertexArray(0);
		glDepthFunc(GL_LESS);
	}

private:
	void SetUp()
	{
		shader = std::make_shared<Shader>("framebuffer_screen");
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(__SCREEN_VERTICES), __SCREEN_VERTICES, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
	}
	unsigned int VAO, VBO;
	std::shared_ptr<Shader> shader;
	std::shared_ptr<glm::vec3> __SCREEN_T = std::make_shared<glm::vec3>(0.89804, 0.69804, 0.68235);
	std::shared_ptr<glm::vec3> __SCREEN_M = std::make_shared<glm::vec3>(0.57647, 0.71273, 0.93333);
	std::shared_ptr<glm::vec3> __SCREEN_B = std::make_shared<glm::vec3>(0.49020, 0.38039, 0.60784);
	float __SCREEN_VERTICES[60] =
	{
		-1.0f, -1.0f, __SCREEN_B->x, __SCREEN_B->y, __SCREEN_B->z,
		1.0f, -1.0f, __SCREEN_B->x, __SCREEN_B->y, __SCREEN_B->z,
		-1.0f,  0.0f, __SCREEN_M->x, __SCREEN_M->y, __SCREEN_M->z,

		-1.0f,  0.0f, __SCREEN_M->x, __SCREEN_M->y, __SCREEN_M->z,
		1.0f, -1.0f, __SCREEN_B->x, __SCREEN_B->y, __SCREEN_B->z,
		1.0f,  0.0f, __SCREEN_M->x, __SCREEN_M->y, __SCREEN_M->z,

		-1.0f,  0.0f, __SCREEN_M->x, __SCREEN_M->y, __SCREEN_M->z,
		1.0f,  0.0f, __SCREEN_M->x, __SCREEN_M->y, __SCREEN_M->z,
		-1.0f,  1.0f, __SCREEN_T->x, __SCREEN_T->y, __SCREEN_T->z,

		-1.0f,  1.0f, __SCREEN_T->x, __SCREEN_T->y, __SCREEN_T->z,
		1.0f,  0.0f, __SCREEN_M->x, __SCREEN_M->y, __SCREEN_M->z,
		1.0f,  1.0f, __SCREEN_T->x, __SCREEN_T->y, __SCREEN_T->z
	};
};

#pragma once
#include <Mesh/MeshBase.h>

class Shader;

class Background : public MeshBase
{
public:
	Background();
	Background(glm::vec3 t, glm::vec3 m, glm::vec3 b);

	void Update(glm::vec3 t, glm::vec3 m, glm::vec3 b);

	void Draw();

private:
	void SetUp();

	unsigned int VAO, VBO;
	std::shared_ptr<Shader> shader;

	glm::vec3 __SCREEN_T = {0.89804, 0.69804, 0.68235};
	glm::vec3 __SCREEN_M = {0.57647, 0.71273, 0.93333};
	glm::vec3 __SCREEN_B = {0.49020, 0.38039, 0.60784};

	float __SCREEN_VERTICES[60] = {
		-1.0f, -1.0f, __SCREEN_B.x, __SCREEN_B.y, __SCREEN_B.z,
		1.0f, -1.0f, __SCREEN_B.x, __SCREEN_B.y, __SCREEN_B.z,
		-1.0f, 0.0f, __SCREEN_M.x, __SCREEN_M.y, __SCREEN_M.z,

		-1.0f, 0.0f, __SCREEN_M.x, __SCREEN_M.y, __SCREEN_M.z,
		1.0f, -1.0f, __SCREEN_B.x, __SCREEN_B.y, __SCREEN_B.z,
		1.0f, 0.0f, __SCREEN_M.x, __SCREEN_M.y, __SCREEN_M.z,

		-1.0f, 0.0f, __SCREEN_M.x, __SCREEN_M.y, __SCREEN_M.z,
		1.0f, 0.0f, __SCREEN_M.x, __SCREEN_M.y, __SCREEN_M.z,
		-1.0f, 1.0f, __SCREEN_T.x, __SCREEN_T.y, __SCREEN_T.z,

		-1.0f, 1.0f, __SCREEN_T.x, __SCREEN_T.y, __SCREEN_T.z,
		1.0f, 0.0f, __SCREEN_M.x, __SCREEN_M.y, __SCREEN_M.z,
		1.0f, 1.0f, __SCREEN_T.x, __SCREEN_T.y, __SCREEN_T.z};
};
